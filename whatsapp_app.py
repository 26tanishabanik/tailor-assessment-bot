import os
import json
import logging
import asyncio
import tempfile
from typing import Dict, List, Any
from datetime import datetime
from dataclasses import dataclass
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from pydub import AudioSegment
from pydub.utils import which
import io

# Main application
from main import Application

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AssessmentResult:
    """Data class for assessment results"""
    image_url: str = None
    audio_url: str = None
    quality_rating: int = 0
    improvement_suggestions: List[str] = None
    stitch_type: str = ""
    technical_issues: List[str] = None
    professional_grade: str = ""
    pass_fail: str = ""
    timestamp: datetime = None
    user_phone: str = ""
    
    def __post_init__(self):
        if self.improvement_suggestions is None:
            self.improvement_suggestions = []
        if self.technical_issues is None:
            self.technical_issues = []
        if self.timestamp is None:
            self.timestamp = datetime.now()

class AudioProcessor:
    """Audio file processing utilities for WhatsApp voice messages"""
    
    def __init__(self):
        """Initialize audio processor"""
        # Check if ffmpeg is available for audio conversion
        if not which("ffmpeg"):
            logger.warning("ffmpeg not found. Audio conversion may fail. Please install it.")
    
    def download_audio_from_twilio(self, media_url: str, account_sid: str, auth_token: str) -> str:
        """
        Download audio file from Twilio URL
        
        Args:
            media_url: Twilio media URL
            account_sid: Twilio account SID
            auth_token: Twilio auth token
            
        Returns:
            Path to downloaded audio file
        """
        try:
            logger.info(f"Downloading audio from: {media_url}")
            
            # Download audio with Twilio authentication
            auth = (account_sid, auth_token)
            response = requests.get(media_url, auth=auth)
            
            if response.status_code != 200:
                raise Exception(f"Failed to download audio. Status: {response.status_code}")
            
            # Save to temporary file with original extension (usually .ogg for WhatsApp)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.ogg')
            temp_file.write(response.content)
            temp_file.close()
            
            logger.info(f"Audio downloaded to: {temp_file.name}")
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Error downloading audio: {e}")
            raise
    
    def convert_ogg_to_wav(self, ogg_file_path: str) -> str:
        """
        Convert OGG audio file to WAV format for ADK
        
        Args:
            ogg_file_path: Path to OGG file
            
        Returns:
            Path to converted WAV file
        """
        try:
            logger.info(f"Converting OGG to WAV: {ogg_file_path}")
            
            # Load OGG file
            audio = AudioSegment.from_ogg(ogg_file_path)
            
            # Convert to WAV with optimal settings for Gemini models
            # Set sample rate to 16kHz and mono channel
            audio = audio.set_frame_rate(16000).set_channels(1)
            
            # Export as WAV
            wav_file_path = ogg_file_path.replace('.ogg', '.wav')
            audio.export(wav_file_path, format="wav")
            
            logger.info(f"Audio converted to WAV: {wav_file_path}")
            return wav_file_path
            
        except Exception as e:
            logger.error(f"Error converting OGG to WAV: {e}")
            raise
    
    def cleanup_temp_files(self, *file_paths: str):
        """Clean up temporary audio files"""
        for file_path in file_paths:
            try:
                if file_path and os.path.exists(file_path):
                    os.unlink(file_path)
                    logger.debug(f"Cleaned up temp file: {file_path}")
            except OSError as e:
                logger.warning(f"Failed to cleanup temp file {file_path}: {e}")

class TwilioWhatsAppAPI:
    """Twilio WhatsApp Business API integration"""
    
    def __init__(self):
        self.account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        self.auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        self.from_number = os.getenv('TWILIO_WHATSAPP_NUMBER', 'whatsapp:+14155238886')  # Default sandbox
        
        # Validate required credentials
        if not all([self.account_sid, self.auth_token]):
            raise ValueError("Missing required Twilio environment variables: TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN")
        
        self.client = Client(self.account_sid, self.auth_token)
        
    def send_message(self, to: str, message: str) -> bool:
        """Send text message via Twilio WhatsApp"""
        try:
            # Ensure both from and to numbers have whatsapp: prefix
            if not to.startswith('whatsapp:'):
                to = f'whatsapp:{to}'
            
            # Ensure from_number has whatsapp: prefix
            from_number = self.from_number
            if not from_number.startswith('whatsapp:'):
                from_number = f'whatsapp:{from_number}'
            
            logger.info(f"Sending message from {from_number} to {to}")
            
            message = self.client.messages.create(
                from_=from_number,
                body=message,
                to=to
            )
            
            logger.info(f"Twilio message sent: {message.sid}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending Twilio WhatsApp message: {str(e)}")
            return False
    
    def send_image_with_caption(self, to: str, image_url: str, caption: str) -> bool:
        """Send image with caption via Twilio WhatsApp"""
        try:
            # Ensure both from and to numbers have whatsapp: prefix
            if not to.startswith('whatsapp:'):
                to = f'whatsapp:{to}'
            
            # Ensure from_number has whatsapp: prefix
            from_number = self.from_number
            if not from_number.startswith('whatsapp:'):
                from_number = f'whatsapp:{from_number}'
            
            message = self.client.messages.create(
                from_=from_number,
                body=caption,
                media_url=[image_url],
                to=to
            )
            
            logger.info(f"Twilio image message sent: {message.sid}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending Twilio WhatsApp image: {str(e)}")
            return False

class WhatsAppSkillAssessmentAgentV3:
    """Enhanced WhatsApp interface with native Gemini audio support"""
    
    def __init__(self):
        # Initialize the proper agent architecture
        self.app_instance = Application()  # Host application with MasterAgent + SubAgents
        self.twilio_api = TwilioWhatsAppAPI()
        self.audio_processor = AudioProcessor()
        self.app = Flask(__name__)
        self.setup_routes()
        
        logger.info("WhatsApp Skill Assessment Agent V3 initialized with native Gemini audio support")
        
    def setup_routes(self):
        """Setup Flask routes for Twilio WhatsApp webhook"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                "status": "healthy", 
                "timestamp": datetime.now().isoformat(),
                "version": "v3",
                "features": ["text", "images", "native_audio"]
            })
        
        @self.app.route('/twilio-webhook', methods=['POST'])
        @self.app.route('/', methods=['POST'])  # Also handle root path
        def handle_twilio_webhook():
            """Handle incoming Twilio WhatsApp messages"""
            try:
                # Get message data from Twilio
                from_number = request.form.get('From', '')
                body = request.form.get('Body', '')
                media_url = request.form.get('MediaUrl0', '')
                media_content_type = request.form.get('MediaContentType0', '')
                num_media = int(request.form.get('NumMedia', '0'))
                
                # Clean phone number (remove whatsapp: prefix) for processing
                phone_number = from_number.replace('whatsapp:', '')
                
                logger.info(f"Twilio webhook: from={from_number} -> {phone_number}, body={body}, media={num_media}, type={media_content_type}")
                
                # Process the message
                if num_media > 0 and media_url:
                    if media_content_type.startswith('image/'):
                        # Handle image message
                        asyncio.run(self.handle_image_message(phone_number, media_url, body))
                    elif media_content_type.startswith('audio/'):
                        # Handle voice message with native Gemini audio
                        asyncio.run(self.handle_voice_message(phone_number, media_url))
                    else:
                        self.send_message(phone_number, "📱 I can process text messages, images, and voice messages for skill assessments.")
                else:
                    # Handle text message
                    asyncio.run(self.handle_text_message(phone_number, body))
                
                # Return TwiML response
                resp = MessagingResponse()
                return str(resp)
                
            except Exception as e:
                logger.error(f"Twilio webhook error: {str(e)}")
                resp = MessagingResponse()
                return str(resp)
    
    def send_message(self, phone_number: str, message: str) -> bool:
        """Send message using Twilio WhatsApp API"""
        return self.twilio_api.send_message(phone_number, message)
    
    async def handle_text_message(self, phone_number: str, text: str):
        """Handle text messages by delegating to Master Agent"""
        try:
            logger.info(f"Processing text from {phone_number}: {text}")
            
            # Pass text directly to Master Agent
            plan = await self.app_instance.master_agent.get_plan(user_query=text)
            
            # Send Master Agent's response to user
            master_response = plan.get('response_to_user', 'I can help you with skill assessment.')
            self.send_message(phone_number, f"🎯 {master_response}")
            
            instructions = plan.get('sub_agent_instructions', [])
            if instructions:
                await self._execute_sub_agents_and_respond(phone_number, instructions)
                
        except Exception as e:
            logger.error(f"Error in Master Agent text processing: {str(e)}")
            self.send_message(phone_number, "❌ I encountered an error. Please try again or send me an image/voice message of your work! 📸🎤")
    
    async def handle_voice_message(self, phone_number: str, media_url: str):
        """
        Handle voice messages by passing the audio data directly to the ADK agent.
        """
        ogg_file_path = None
        wav_file_path = None
        
        try:
            logger.info(f"Processing voice message from {phone_number}")
            self.send_message(phone_number, "🎤 Processing your voice message with native audio analysis...")
            
            # **STEP 1: Download and convert audio file**
            ogg_file_path = self.audio_processor.download_audio_from_twilio(
                media_url, 
                self.twilio_api.account_sid, 
                self.twilio_api.auth_token
            )
            wav_file_path = self.audio_processor.convert_ogg_to_wav(ogg_file_path)
            
            # **STEP 2: Read audio data and pass to ADK agent**
            self.send_message(phone_number, "🧠 Analyzing audio with Master Agent...")
            
            with open(wav_file_path, "rb") as f:
                audio_data = f.read()

            # Pass audio data directly to the Master Agent
            # The user_query can be a placeholder as the model will interpret the audio
            plan = await self.app_instance.master_agent.get_plan(user_query="voice message", audio_data=audio_data)

            master_response = plan.get('response_to_user', 'I can help with skill assessment.')
            self.send_message(phone_number, f"🎯 {master_response}")
            
            instructions = plan.get('sub_agent_instructions', [])
            if instructions:
                await self._execute_sub_agents_and_respond(phone_number, instructions)
                
        except Exception as e:
            logger.error(f"Error processing voice message from {phone_number}: {str(e)}")
            self.send_message(phone_number, f"❌ An error occurred with native audio analysis: {str(e)}. Please try again.")
            
        finally:
            self.audio_processor.cleanup_temp_files(ogg_file_path, wav_file_path)

    async def handle_image_message(self, phone_number: str, media_url: str, caption: str = ""):
        """Handle image messages by delegating to Master Agent and Sub-Agents"""
        try:
            logger.info(f"Processing image from {phone_number} with caption: {caption}")
            
            self.send_message(phone_number, "📥 Downloading your image...")
            auth = (self.twilio_api.account_sid, self.twilio_api.auth_token)
            response = requests.get(media_url, auth=auth)
            
            if response.status_code != 200:
                logger.error(f"Failed to download image: Status {response.status_code}")
                self.send_message(phone_number, "❌ Sorry, I couldn't download your image. Please try again.")
                return
            
            image_bytes = response.content
            
            # Master Agent Planning
            user_query = caption if caption.strip() else "Here is my work sample for job skill assessment"
            self.send_message(phone_number, "🧠 Master Agent analyzing your request...")
            
            # Pass image data directly to the Master Agent
            plan = await self.app_instance.master_agent.get_plan(user_query=user_query, image_data=image_bytes)
            
            master_response = plan.get('response_to_user', 'Processing your skill assessment...')
            self.send_message(phone_number, f"🎯 Master Agent: {master_response}")
            
            instructions = plan.get('sub_agent_instructions', [])
            if instructions:
                await self._execute_sub_agents_and_respond(phone_number, instructions)
            else:
                logger.info("No sub-agent instructions from Master Agent")
            
        except Exception as e:
            logger.error(f"Error in Master Agent image processing: {str(e)}")
            self.send_message(phone_number, f"❌ Master Agent error: {str(e)}. Please try again.")

    async def _execute_sub_agents_and_respond(self, phone_number: str, instructions: List[Dict]):
        """Execute sub-agent instructions and get final verdict from Master Agent"""
        try:
            self.send_message(phone_number, "🔧 Executing specialized sub-agents...")
            
            sub_agent_results = []
            for instruction in instructions:
                agent_name = instruction.get('agent_name', 'Unknown')
                skill = instruction.get('task_context', {}).get('skill_to_assess', 'Unknown')
                
                self.send_message(phone_number, f"🤖 {agent_name} analyzing {skill}...")
                
                # Execute sub-agent via host application
                # Note: For stitching_assessor_agent, it will still expect image_data
                # If the original input was audio, you'd need to decide how to handle
                # passing a relevant image to the sub-agent, or modify sub-agent logic.
                result = await self.app_instance.execute_sub_agent_instruction(instruction)
                
                sub_agent_results.append({
                    "agent_name": agent_name,
                    "skill_assessed": skill,
                    "result": result,
                })
            
            self.send_message(phone_number, "⚖️ Master Agent making final decision...")
            
            target_role = instructions[0].get('task_context', {}).get('role', 'Unknown')
            final_verdict = await self.app_instance.master_agent.get_final_verdict(target_role, sub_agent_results)
            
            await self._send_agent_results(phone_number, sub_agent_results, final_verdict)
            
        except Exception as e:
            logger.error(f"Error executing sub-agents: {str(e)}")
            self.send_message(phone_number, f"❌ Sub-agent execution error: {str(e)}")

    async def _send_agent_results(self, phone_number: str, sub_agent_results: List[Dict], final_verdict: Dict):
        """Send formatted results from the agent system"""
        try:
            final_response = final_verdict.get('response_to_user', 'Assessment completed.')
            self.send_message(phone_number, f"🏁 **Final Assessment:**\n{final_response}")
            
            if sub_agent_results and sub_agent_results[0].get('result', {}).get('data'):
                await self._send_detailed_assessment(phone_number, sub_agent_results[0]['result']['data'])
            
            decision_data = final_verdict.get('final_decision_data', {})
            decision = decision_data.get('decision', 'UNKNOWN')
            decision_emoji = "✅" if decision == "PASS" else "❌" if decision == "FAIL" else "⏳"
            
            self.send_message(phone_number, f"{decision_emoji} **Final Result: {decision}**")
            
        except Exception as e:
            logger.error(f"Error sending agent results: {str(e)}")
            self.send_message(phone_number, "Assessment completed but there was an error formatting the results.")

    async def _send_detailed_assessment(self, phone_number: str, assessment_data: Dict):
        """Send detailed assessment data from sub-agents"""
        try:
            rating = assessment_data.get('quality_rating', 5)
            rating_emoji = "🔴" if rating < 3 else "🟠" if rating < 5 else "🟡" if rating < 7 else "🟢" if rating < 9 else "🏆"
            
            details = f"""📊 **Detailed Analysis:**

{rating_emoji} **Quality Rating:** {rating}/10
📝 **Stitch Type:** {assessment_data.get('stitch_type', 'Unknown')}
🎯 **Skill Level:** {assessment_data.get('professional_grade', 'Unknown').title()}

⚠️ **Technical Issues:**
{self._format_list(assessment_data.get('technical_issues', []))}

💡 **Improvement Tips:**
{self._format_list(assessment_data.get('improvement_suggestions', []))}"""
            
            self.send_message(phone_number, details)
            
        except Exception as e:
            logger.error(f"Error sending detailed assessment: {str(e)}")

    def _format_list(self, items: List[str]) -> str:
        """Format list items with bullet points"""
        if not items:
            return "• None identified"
        return "\n".join(f"• {item}" for item in items[:3])
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask application"""
        logger.info(f"Starting WhatsApp Skill Assessment Agent V3 on {host}:{port}")
        logger.info(f"Webhook URL: http://{host}:{port}/twilio-webhook")
        logger.info("Features: Text, Images, Native Gemini Audio")
        self.app.run(host=host, port=port, debug=debug)

def main():
    """Main function to run the WhatsApp Skill Assessment Agent V3"""
    try:
        agent = WhatsAppSkillAssessmentAgentV3()
        
        port = int(os.getenv('PORT', 5000))
        agent.run(host='0.0.0.0', port=port, debug=False)
        
    except Exception as e:
        logger.error(f"Failed to start WhatsApp Skill Assessment Agent V3: {str(e)}")
        print(f"Error: {str(e)}")
        print("\nRequired environment variables:")
        print("- TWILIO_ACCOUNT_SID")
        print("- TWILIO_AUTH_TOKEN") 
        print("- TWILIO_WHATSAPP_NUMBER (optional, defaults to sandbox)")
        print("- GEMINI_API_KEY") # Still needed for ADK
        # GOOGLE_APPLICATION_CREDENTIALS is NOT needed for native Gemini audio,
        # but might be for other Google Cloud services if you use them.

if __name__ == "__main__":
    main()