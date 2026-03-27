
from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)
_MAX_SCRIPT_RETRIES = 3

class ScriptWriter:
    """
    Video Agent Sub-module:
    Generates lecture scripts for each slide image.
    Maintains a 'Context Chain' to ensure continuity.
    """
    
    def __init__(self, client: OpenAI):
        self.client = client
        self.context_summary = "Opening of the lecture."

    def generate_scripts(self, image_paths: List[Path], topic: str) -> List[str]:
        """
        Iterate through slide images and generate a script for each.
        Returns a list of script strings (narration).
        """
        scripts = []
        
        logger.info(f"Generating scripts for {len(image_paths)} slides based on topic: {topic}")
        
        for i, img_path in enumerate(image_paths):
            script = self._generate_single_script(img_path, i, len(image_paths), topic)
            scripts.append(script)
            
        return scripts

    def _generate_single_script(self, img_path: Path, index: int, total: int, topic: str) -> str:
        """
        Call GPT-4o Vision for one slide.
        """
        # Encode image
        with open(img_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        # Construct Prompt with Context Chain
        is_first = (index == 0)
        is_last = (index == total - 1)
        
        prompt = f"""
        You are a professional lecturer teaching a course on: "{topic}".
        
        Current Status: Slide {index + 1} of {total}.
        Previous Context: "{self.context_summary}"
        
        Task:
        1. Analyze the visual content of the attached slide (image).
        2. Write a natural, engaging lecture script for this specific slide.
        3. Ensure smooth transition from the previous context.
        4. { "This is the first slide, so start with a welcoming intro." if is_first else "" }
        5. { "This is the last slide, so conclude the lecture." if is_last else "" }
        
        Output Format:
        Return ONLY the spoken narration text. No "Slide 1:" prefixes or markdown.
        """
        
        last_error: Optional[Exception] = None
        for attempt in range(1, _MAX_SCRIPT_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o", 
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}", 
                                        "detail": "high" # Use high detail for text legibility
                                    }
                                },
                            ],
                        }
                    ],
                    max_tokens=500
                )
                
                script_text = response.choices[0].message.content.strip()
                if not script_text:
                    raise ValueError("Empty narration returned by model.")
                
                # Update Context for next slide (Self-Correction/Summarization)
                self._update_context(script_text)
                
                logger.info(f"Generated script for Slide {index+1}: {script_text[:50]}...")
                return script_text
                
            except Exception as e:
                last_error = e
                logger.warning(
                    "Failed to generate script for %s on attempt %d/%d: %s",
                    img_path,
                    attempt,
                    _MAX_SCRIPT_RETRIES,
                    e,
                )

        raise RuntimeError(f"Failed to generate script for {img_path}: {last_error}")

    def _update_context(self, current_script: str):
        """
        Update the running summary of what has been said.
        Simple approach: Keep the last 2 sentences or generate a quick summary.
        For speed, we just truncate and keep the trailing part to establish flow.
        """
        # Ideally we'd ask LLM to summarize, but that doubles cost.
        # Simple heuristic: "Just covered: [last 100 chars]"
        self.context_summary = f"Just discussed: ...{current_script[-200:] if len(current_script) > 200 else current_script}"
