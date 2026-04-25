"""
agent/rag_pipeline.py
RAG (Retrieval-Augmented Generation) pipeline for AutoStream knowledge base.
Uses a local JSON knowledge base with semantic chunking and keyword matching.
"""

import json
import os
import re
from typing import List, Dict, Any


class AutoStreamRAG:
    """
    Simple but effective RAG pipeline that:
    1. Loads structured knowledge from JSON
    2. Chunks it into searchable passages
    3. Retrieves the most relevant chunks for a query
    """

    def __init__(self, kb_path: str = None):
        if kb_path is None:
            kb_path = os.path.join(
                os.path.dirname(__file__), "..", "knowledge_base", "autostream_kb.json"
            )
        self.kb_path = kb_path
        self.knowledge_base = self._load_kb()
        self.chunks = self._build_chunks()

    def _load_kb(self) -> Dict[str, Any]:
        """Load knowledge base from JSON file."""
        with open(self.kb_path, "r") as f:
            return json.load(f)

    def _build_chunks(self) -> List[Dict[str, Any]]:
        """
        Convert structured JSON KB into searchable text chunks with metadata.
        Each chunk has: text, keywords, category
        """
        chunks = []
        kb = self.knowledge_base

        # --- Company Overview ---
        chunks.append({
            "category": "company",
            "keywords": ["autostream", "company", "what is", "about", "product", "platform"],
            "text": (
                f"AutoStream is a SaaS platform — {kb['company']['tagline']}. "
                f"{kb['company']['description']}"
            )
        })

        # --- Pricing Overview ---
        chunks.append({
            "category": "pricing",
            "keywords": ["price", "pricing", "cost", "plan", "how much", "pay", "subscription", "fee"],
            "text": (
                "AutoStream offers two plans:\n"
                "• Basic Plan: $29/month — 10 videos/month, 720p resolution export.\n"
                "• Pro Plan: $79/month — Unlimited videos, 4K resolution, AI captions, priority 24/7 support, "
                "analytics, and multi-platform publishing.\n"
                "Annual billing is also available at a discounted rate."
            )
        })

        # --- Basic Plan Detail ---
        basic = kb["pricing"]["plans"][0]
        chunks.append({
            "category": "pricing_basic",
            "keywords": ["basic", "basic plan", "cheap", "starter", "$29", "29"],
            "text": (
                f"Basic Plan costs ${basic['price_monthly']}/month (${basic['price_annual']}/year). "
                f"It includes: {', '.join(basic['features'])}. "
                f"Best for: {basic['best_for']}."
            )
        })

        # --- Pro Plan Detail ---
        pro = kb["pricing"]["plans"][1]
        chunks.append({
            "category": "pricing_pro",
            "keywords": ["pro", "pro plan", "unlimited", "4k", "captions", "professional", "$79", "79", "premium"],
            "text": (
                f"Pro Plan costs ${pro['price_monthly']}/month (${pro['price_annual']}/year). "
                f"It includes: {', '.join(pro['features'])}. "
                f"Best for: {pro['best_for']}."
            )
        })

        # --- Refund Policy ---
        refund = kb["policies"]["refund"]
        chunks.append({
            "category": "policy_refund",
            "keywords": ["refund", "money back", "return", "cancel", "cancellation", "guarantee"],
            "text": (
                f"Refund Policy: {refund['policy']} {refund['details']}"
            )
        })

        # --- Support Policy ---
        support = kb["policies"]["support"]
        chunks.append({
            "category": "policy_support",
            "keywords": ["support", "help", "customer service", "contact", "24/7", "response"],
            "text": (
                f"Support Policy: Basic plan users get {support['basic_plan']} "
                f"Pro plan users get {support['pro_plan']} "
                f"Note: {support['note']}"
            )
        })

        # --- Free Trial ---
        trial = kb["policies"]["trial"]
        chunks.append({
            "category": "trial",
            "keywords": ["trial", "free", "try", "test", "demo", "no credit card"],
            "text": (
                f"Free Trial: {trial['details']} Duration: {trial['duration']}."
            )
        })

        # --- Cancellation ---
        cancel = kb["policies"]["cancellation"]
        chunks.append({
            "category": "cancellation",
            "keywords": ["cancel", "cancellation", "stop", "end subscription"],
            "text": f"Cancellation Policy: {cancel['policy']}"
        })

        # --- Features ---
        features = kb["features_overview"]
        chunks.append({
            "category": "features",
            "keywords": ["feature", "features", "ai", "caption", "template", "edit", "collaboration", "auto"],
            "text": (
                "AutoStream Key Features:\n"
                f"• AI Captions: {features['ai_captions']}\n"
                f"• Auto Editing: {features['auto_editing']}\n"
                f"• Templates: {features['templates']}\n"
                f"• Collaboration: {features['collaboration']}"
            )
        })

        # --- FAQ Chunks ---
        for faq in kb["faq"]:
            keywords = self._extract_keywords(faq["question"] + " " + faq["answer"])
            chunks.append({
                "category": "faq",
                "keywords": keywords,
                "text": f"Q: {faq['question']}\nA: {faq['answer']}"
            })

        return chunks

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract simple keywords from text."""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        stopwords = {"the", "and", "for", "are", "that", "this", "with", "from", "your", "can", "you", "our", "all"}
        return [w for w in words if w not in stopwords]

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant knowledge chunks for a query.
        Uses keyword overlap scoring.
        
        Args:
            query: User's question
            top_k: Number of chunks to return
            
        Returns:
            List of relevant chunks sorted by relevance score
        """
        query_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))
        
        scored_chunks = []
        for chunk in self.chunks:
            keyword_set = set(chunk["keywords"])
            overlap = len(query_words.intersection(keyword_set))
            
            # Boost score if query words appear in the chunk text directly
            text_matches = sum(1 for word in query_words if word in chunk["text"].lower())
            score = overlap * 2 + text_matches
            
            if score > 0:
                scored_chunks.append((score, chunk))

        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        return [chunk for _, chunk in scored_chunks[:top_k]]

    def get_context_string(self, query: str) -> str:
        """
        Returns a formatted string of relevant knowledge for injection into LLM prompt.
        """
        relevant_chunks = self.retrieve(query, top_k=3)
        
        if not relevant_chunks:
            return "No specific information found in knowledge base for this query."
        
        context_parts = []
        for i, chunk in enumerate(relevant_chunks, 1):
            context_parts.append(f"[Knowledge {i} - {chunk['category'].upper()}]\n{chunk['text']}")
        
        return "\n\n".join(context_parts)
