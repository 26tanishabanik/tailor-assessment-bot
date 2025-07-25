import os
import json
import logging
import asyncio
import base64
from typing import Dict, List, Any
from datetime import datetime
from dataclasses import dataclass
from PIL import Image
import io
import os
import requests
from flask import Flask, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

@dataclass
class StitchingAssessment:
    """Data class for stitching assessment results"""
    image_url: str
    quality_rating: int
    improvement_suggestions: List[str]
    stitch_type: str
    technical_issues: List[str]
    professional_grade: str
    pass_fail: str
    timestamp: datetime
    user_phone: str

class GeminiStitchingExpert:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.expert_prompt = self._create_expert_prompt()
        
    def _create_expert_prompt(self) -> str:
        """Create the expert tailor prompt for Gemini"""
        return """
        You are someone who has gone through a tailoring course and trying to start a career.  You have learnt various traditional hand-stitching techniques, machine operations, and quality control.
        Your task is to stitch a 100 1 cm by 1cm squares on a 10 cm by 10 cm cloth 
        PASS CRITERIA: 
        a. 70% or more of the squares actually being squares 
        b. 70% of the lines (10 horizontal parallel lines, 10 vertical parallel lines) being parallel to each other 
        ASSESSMENT CRITERIA:
        
          1. STITCH QUALITY (1-10 scale):  
          - Evenness and consistency of stitches  
          - Thread tension (too tight/loose)  
          - Stitch length uniformity  
          - Seam alignment and straightness  

          2. TECHNICAL EXECUTION:  
          - Proper seam allowances  
          - Correct stitch type for fabric  
          - Finishing techniques  
          - Professional appearance
          
          3. STRUCTURAL INTEGRITY:  
          - Seam strength  
          - Durability assessment  
          - Stress point reinforcement  
          - Overall construction quality  

          4. PROFESSIONAL STANDARDS:
          - Industry acceptability
          - Commercial viability
          - Craftsmanship level
          - Attention to detail 
  
          RATING SCALE:  
          - 1-2: Beginner level, major issues
          - 3-4: Novice, needs significant improvement
          - 5-6: Intermediate, acceptable for practice  
          - 7-8: Advanced, good quality work
          - 9-10: Expert level, professional standard
        
        RESPONSE FORMAT:
        Return a JSON object with:
        {
            "quality_rating": <1-10>,
            "stitch_type": "<identified stitch type>",
            "technical_issues": ["<max 2 key issues>"],
            "improvement_suggestions": ["<max 2 actionable tips>"],
            "professional_grade": "<beginner/novice/intermediate/advanced/expert>",
            "pass_fail": "<pass|fail>"
        }

        Keep detailed_analysis under 150 characters and each issue/suggestion under 50 characters.
            
        Analyze the stitching with the precision of a professional tailor evaluating work for a medium size fashion house.
        """
    
    async def analyze_stitching(self, image_data: bytes) -> Dict[str, Any]:
        try:
            image = Image.open(io.BytesIO(image_data))
            image_parts = [
                {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(image_data).decode('utf-8')
                }
            ]
            response = self.model.generate_content([
                self.expert_prompt,
                image_parts[0]
            ])
            try:
                response_text = response.text
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    if json_end != -1:
                        response_text = response_text[json_start:json_end].strip()
                
                result = json.loads(response_text)
                return result
            except json.JSONDecodeError:
                return self._parse_fallback_response(response.text)
                
        except Exception as e:
            logger.error(f"Error analyzing stitching: {str(e)}")
            return {
                "quality_rating": 5,
                "stitch_type": "unknown",
                "technical_issues": ["Analysis failed"],
                "improvement_suggestions": ["Please try again with a clearer image"],
                "professional_grade": "unknown",
                "pass_fail": "fail"
            }
    
    def _parse_fallback_response(self, response_text: str) -> Dict[str, Any]:
        rating = 5
        import re
        rating_match = re.search(r'rating[:\s]*(\d+)', response_text, re.IGNORECASE)
        if rating_match:
            rating = int(rating_match.group(1))
        
        return {
            "quality_rating": rating,
            "stitch_type": "general",
            "technical_issues": ["Could not parse detailed issues"],
            "improvement_suggestions": ["Please refer to the detailed analysis"],
            "professional_grade": "intermediate",
            "pass_fail": "fail" if rating < 7 else "pass"
        }

class TwilioWhatsAppAPI:
    def __init__(self):
        self.account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        self.auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        self.from_number = os.getenv('TWILIO_WHATSAPP_NUMBER')
        
        if not all([self.account_sid, self.auth_token]):
            raise ValueError("Missing required Twilio environment variables: TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN")
        
        self.client = Client(self.account_sid, self.auth_token)
        
    def send_message(self, to: str, message: str) -> bool:
        try:
            # Ensure phone number is in correct format
            if not to.startswith('whatsapp:'):
                to = f'whatsapp:{to}'
            
            message = self.client.messages.create(
                from_=self.from_number,
                body=message,
                to=to
            )
            
            logger.info(f"Twilio message sent: {message.sid}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending Twilio WhatsApp message: {str(e)}")
            return False
    
    def send_image_with_caption(self, to: str, image_url: str, caption: str) -> bool:
        try:
            # Ensure phone number is in correct format
            if not to.startswith('whatsapp:'):
                to = f'whatsapp:{to}'
            
            message = self.client.messages.create(
                from_=self.from_number,
                body=caption,
                media_url=[image_url],
                to=to
            )
            
            logger.info(f"Twilio image message sent: {message.sid}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending Twilio WhatsApp image: {str(e)}")
            return False

class MultimodalStitchingAgent:
    def __init__(self):
        self.gemini_expert = GeminiStitchingExpert()
        self.twilio_api = TwilioWhatsAppAPI()
        self.app = Flask(__name__)
        self.setup_routes()
        
        logger.info("Using Twilio WhatsApp Business Number")
        
    def setup_routes(self):
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})
        
        @self.app.route('/twilio-webhook', methods=['POST'])
        def handle_twilio_webhook():
            try:
                from_number = request.form.get('From', '')
                body = request.form.get('Body', '')
                media_url = request.form.get('MediaUrl0', '')
                num_media = int(request.form.get('NumMedia', '0'))
                
                phone_number = from_number.replace('whatsapp:', '')
                
                logger.info(f"Twilio webhook: from={phone_number}, body={body}, media={num_media}")
                
                if num_media > 0 and media_url:
                    asyncio.run(self.handle_twilio_image_message(phone_number, media_url))
                else:
                    asyncio.run(self.handle_text_message(phone_number, body))
                
                resp = MessagingResponse()
                return str(resp)
                
            except Exception as e:
                logger.error(f"Twilio webhook error: {str(e)}")
                resp = MessagingResponse()
                return str(resp)
    
    def send_message(self, phone_number: str, message: str) -> bool:
        return self.twilio_api.send_message(phone_number, message)
    
    async def handle_text_message(self, phone_number: str, text: str):
        text_lower = text.lower()
        
        if any(greeting in text_lower for greeting in ['hi', 'hello', 'start', 'help']):
            welcome_msg = """
🧵 Welcome to the Professional Stitching Quality Assessment Bot! 

I'm your AI tailor expert with 30+ years of experience. I can analyze your stitching work and provide professional feedback.

📸 **How to use:**
1. Send me a clear photo of your stitching work
2. I'll analyze it like a professional tailor
3. Get detailed feedback with ratings (1-10)
4. Receive improvement suggestions

**What I can assess:**
• Hand stitching quality
• Machine stitch precision
• Seam construction
• Thread tension
• Professional finishing
• Overall craftsmanship

Send me your stitching photo to get started! 🪡
            """
            self.send_message(phone_number, welcome_msg)
            
        else:
            self.send_message(
                phone_number, 
                "📸 Please send me a clear photo of your stitching work for professional assessment!"
            )
    
    async def handle_twilio_image_message(self, phone_number: str, media_url: str):
        try:
            self.send_message(
                phone_number,
                "🔍 Analyzing your stitching work with professional expertise... Please wait."
            )
            
            auth = (self.twilio_api.account_sid, self.twilio_api.auth_token)
            response = requests.get(media_url, auth=auth)
            
            if response.status_code != 200:
                logger.error(f"Failed to download image: Status {response.status_code}, URL: {media_url}")
                self.send_message(
                    phone_number,
                    "❌ Sorry, I couldn't download your image. Please try again."
                )
                return
            
            image_bytes = response.content
            logger.info(f"Successfully downloaded image: {len(image_bytes)} bytes")
            
            analysis = await self.gemini_expert.analyze_stitching(image_bytes)
            
            assessment = StitchingAssessment(
                image_url=media_url,
                quality_rating=analysis['quality_rating'],
                improvement_suggestions=analysis['improvement_suggestions'],
                stitch_type=analysis['stitch_type'],
                technical_issues=analysis['technical_issues'],
                professional_grade=analysis['professional_grade'],
                pass_fail=analysis.get('pass_fail', 'fail'),
                timestamp=datetime.now(),
                user_phone=phone_number
            )
            
            await self.send_assessment_report(phone_number, assessment)
            
        except Exception as e:
            logger.error(f"Error processing Twilio image: {str(e)}")
            self.send_message(
                phone_number,
                f"❌ Error analyzing your stitching: {str(e)}. Please try again."
            )
    

    async def send_assessment_report(self, phone_number: str, assessment: StitchingAssessment):
        rating_emoji = {
            range(1, 3): "🔴",
            range(3, 5): "🟠", 
            range(5, 7): "🟡",
            range(7, 9): "🟢",
            range(9, 11): "🏆"
        }
        
        emoji = "🟡"  # default
        for score_range, emo in rating_emoji.items():
            if assessment.quality_rating in score_range:
                emoji = emo
                break
        
        pass_fail_emoji = "✅" if assessment.pass_fail.lower() == "pass" else "❌"
        
        report = f"""🧵 STITCHING ASSESSMENT 🧵

{emoji} Rating: {assessment.quality_rating}/10
{pass_fail_emoji} Result: {assessment.pass_fail.upper()}
📝 Type: {assessment.stitch_type.title()}
🎯 Level: {assessment.professional_grade.title()}

⚠️ Issues: {self._format_list_compact(assessment.technical_issues)}

💡 Tips: {self._format_list_compact(assessment.improvement_suggestions)}

🎓 Feedback: {self._get_professional_feedback_short(assessment.quality_rating)}"""
        
        self.send_message(phone_number, report)
        
        # Send concise follow-up encouragement
        if assessment.quality_rating >= 8:
            self.send_message(phone_number, "🌟 Excellent professional quality work!")
        elif assessment.quality_rating >= 6:
            self.send_message(phone_number, "👍 Good progress! Focus on suggestions to reach pro level.")
        else:
            self.send_message(phone_number, "💪 Keep practicing! Focus on fundamentals.")
    
    def _format_list(self, items: List[str]) -> str:
        if not items:
            return "• None identified"
        return "\n".join(f"• {item}" for item in items)
    
    def _format_list_compact(self, items: List[str]) -> str:
        if not items:
            return "None"
        return " | ".join(items[:2])
    
    def _get_professional_feedback(self, rating: int) -> str:
        feedback_map = {
            range(1, 3): "Beginner level - Focus on basic techniques and hand positioning. Practice with simple straight lines.",
            range(3, 5): "Developing skills - Work on consistency and tension control. Consider taking a basic tailoring course.",
            range(5, 7): "Intermediate level - Good foundation! Focus on precision and finishing techniques.",
            range(7, 9): "Advanced work - Professional quality emerging. Refine details for commercial standards.",
            range(9, 11): "Expert level - Outstanding craftsmanship! This meets high-end commercial standards."
        }
        
        for score_range, feedback in feedback_map.items():
            if rating in score_range:
                return feedback
        
        return "Continue practicing to develop your skills!"
    
    def _get_professional_feedback_short(self, rating: int) -> str:
        feedback_map = {
            range(1, 3): "Beginner - Practice basic techniques",
            range(3, 5): "Developing - Work on consistency", 
            range(5, 7): "Intermediate - Good foundation!",
            range(7, 9): "Advanced - Professional quality emerging",
            range(9, 11): "Expert - Outstanding craftsmanship!"
        }
        
        for score_range, feedback in feedback_map.items():
            if rating in score_range:
                return feedback
        
        return "Keep practicing!"
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        logger.info(f"Starting Multimodal Stitching Agent on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

def main():
    agent = MultimodalStitchingAgent()
    port = int(os.getenv('PORT', 8000))
    agent.run(host='0.0.0.0', port=port, debug=False)

if __name__ == "__main__":
    main()
