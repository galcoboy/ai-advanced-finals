"""LangChain chatbot with Gemini — menu, hours, reservations, and n8n webhooks."""

import json
import os
import re
from typing import List

import requests
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from restaurant_db import (
    book_reservation,
    cancel_reservation,
    get_restaurant_details_and_hours,
    search_menu_items,
)


class RestaurantChatbot:
    """RAG-style restaurant assistant backed by SQLite tables."""

    def __init__(self, db_path: str, model_name: str = "gemini-flash-latest") -> None:
        self.db_path = db_path
        self.llm = None

        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=api_key)

        self.classifier_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a router for a restaurant chatbot. "
                    "Classify the user message into exactly one of these categories:\n"
                    " reservation — user wants to book a table\n"
                    " cancellation — user wants to cancel an existing booking\n"
                    " menu — questions about food, drinks, or prices\n"
                    " hours — questions about opening hours or location\n"
                    " general — anything else\n"
                    "Return ONLY the single category word. No punctuation, no explanation.",
                ),
                ("human", "{question}"),
            ]
        )

        self.answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful restaurant assistant. Use only the provided context. "
                    "If context does not contain the answer, say you are not sure and ask a clarifying question. "
                    "Never ask which restaurant the user means.",
                ),
                ("human", "Question: {question}\n\nContext:\n{context}"),
            ]
        )

    def classify_question(self, question: str) -> str:
        """Use the LLM to classify intent, with keyword fallbacks when no API key."""
        lower_q = question.lower()

        if any(k in lower_q for k in ["cancel", "cancellation"]):
            return "cancellation"
        if any(k in lower_q for k in ["book", "reserve", "reservation", "table for"]):
            return "reservation"
        if any(
            k in lower_q
            for k in ["menu", "dish", "food", "price", "vegan", "vegetarian", "spicy", "drink"]
        ):
            return "menu"
        if any(
            k in lower_q
            for k in ["hour", "open", "close", "address", "phone", "location", "email", "website"]
        ):
            return "hours"

        if not self.llm:
            return "general"

        chain = self.classifier_prompt | self.llm | StrOutputParser()
        result = chain.invoke({"question": question}).strip().lower()
        valid = {"reservation", "cancellation", "menu", "hours", "general"}
        return result if result in valid else "general"

    def _build_menu_context(self, question: str) -> tuple[str, bool]:
        rows = search_menu_items(self.db_path, question)
        if not rows:
            return "No menu records matched the question.", False

        lines: List[str] = []
        for row in rows:
            veg = "vegetarian" if row["is_vegetarian"] else "non-vegetarian"
            spicy = "spicy" if row["is_spicy"] else "not spicy"
            status = "available" if row["is_available"] else "currently unavailable"
            lines.append(
                f"- {row['item_name']} ({row['category']}): {row['description']} | "
                f"${row['price']:.2f} | {veg}, {spicy}, {status}"
            )
        return "\n".join(lines), True

    def _build_details_context(self) -> str:
        details, hours = get_restaurant_details_and_hours(self.db_path)
        if not details:
            return "No restaurant details found."

        details_text = (
            f"Name: {details['name']}\n"
            f"Address: {details['address']}\n"
            f"Phone: {details['phone']}\n"
            f"Email: {details['email']}\n"
            f"Website: {details['website']}"
        )

        hours_lines = [
            f"- {h['day_of_week']}: {h['open_time']} to {h['close_time']}"
            + (f" ({h['notes']})" if h.get("notes") else "")
            for h in hours
        ]
        return details_text + "\n\nOpening Hours:\n" + "\n".join(hours_lines)

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Parse JSON from LLM output, stripping markdown fences if present."""
        text = raw.strip()
        if text.startswith("```"):
            text = text.removeprefix("```json").removeprefix("```").strip()
            if text.endswith("```"):
                text = text[:-3].strip()
        return json.loads(text)

    def _handle_reservation(self, question: str) -> str:
        extract_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Extract reservation details from the message. "
                    "Return ONLY valid JSON with keys: "
                    "customer_name, date, time, party_size, contact. "
                    "Use null for missing fields. No explanation.",
                ),
                ("human", "{question}"),
            ]
        )

        if not self.llm:
            return "Please call us directly to make a reservation!"

        chain = extract_prompt | self.llm | StrOutputParser()
        raw = chain.invoke({"question": question})
        try:
            details = self._parse_json(raw)
            required = ["customer_name", "date", "time", "party_size"]
            if not all(details.get(k) for k in required):
                return (
                    "I need your name, date, time, and party size. "
                    "Example: 'Table for 2 on Friday at 7pm, name is Sara'"
                )

            res_id = book_reservation(
                self.db_path,
                details["customer_name"],
                details["date"],
                str(details["time"]),
                int(details["party_size"]),
                details.get("contact"),
            )
            self._notify_n8n({**details, "id": res_id}, event="reservation")
            return (
                f"✅ Reservation confirmed!\n"
                f"Name: {details['customer_name']}\n"
                f"Date: {details['date']} at {details['time']}\n"
                f"Party of {details['party_size']} · Booking #{res_id}"
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            return "Sorry, I couldn't process that. Please try again."

    def _handle_cancellation(self, question: str) -> str:
        match = re.search(r"\b(\d+)\b", question)
        if match:
            res_id = int(match.group(1))
            cancel_reservation(self.db_path, res_id)
            self._notify_n8n({"id": res_id}, event="cancellation")
            return f"Reservation #{res_id} has been cancelled."
        return "Please provide your booking ID number to cancel."

    def _notify_n8n(self, data: dict, event: str) -> None:
        """Fire-and-forget webhook to n8n. Never crashes the chatbot."""
        webhook_url = os.getenv("N8N_WEBHOOK_URL")
        if not webhook_url:
            return
        try:
            requests.post(webhook_url, json={**data, "event": event}, timeout=5)
        except Exception:
            pass

    def answer(self, question: str) -> str:
        """Route question, retrieve matching SQLite data, and generate an answer."""
        category = self.classify_question(question)

        if category == "reservation":
            return self._handle_reservation(question)
        if category == "cancellation":
            return self._handle_cancellation(question)

        if category == "menu":
            context, has_match = self._build_menu_context(question)
            if not has_match:
                return (
                    "I could not find that item in the current menu. "
                    "Ask me to list available mains, starters, desserts, or drinks."
                )
        elif category == "hours":
            context = self._build_details_context()
        else:
            return (
                "I can help with menu items, prices, table reservations, cancellations, "
                "and restaurant details like opening hours, phone, and address."
            )

        if not self.llm:
            return f"(Local fallback, no Gemini API key configured)\n{context}"

        chain = self.answer_prompt | self.llm | StrOutputParser()
        return chain.invoke({"question": question, "context": context})
