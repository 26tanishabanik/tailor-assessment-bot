import os
import asyncio
import logging
import requests
import base64
from flask import Flask, request, Response
from twilio.twiml.messaging_response import MessagingResponse
from multimodal_adk_agent import MultimodalADKAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Verify required environment variables
api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found in environment variables.")

required_env_vars = ["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"]
for var in required_env_vars:
    if not os.getenv(var):
        raise ValueError(f"{var} not found in environment variables.")

# --- 1. Initialize the ADK Agent with Runner and SessionService ---
# Create the ADK custom agent instance following the example pattern
multimodal_agent = MultimodalADKAgent(name="GoogleAskAgent", prompts_dir="../agent_prompts")

# Initialize Runner and SessionService like in the ADK example
session_service = InMemorySessionService()
runner = Runner(
    agent=multimodal_agent,
    app_name="multimodal_assessment_app", 
    session_service=session_service
)

# --- 2. Flask Web Server ---
app = Flask(__name__)

@app.route("/twilio-webhook", methods=['POST'])
def twilio_webhook():
    """
    Twilio webhook handler that uses the ADK MultimodalADKAgent for processing.
    The agent automatically handles both text and audio inputs using its _run_async_impl method.
    """
    twilio_response = MessagingResponse()
    media_url = request.values.get('MediaUrl0', None)
    media_type = request.values.get('MediaContentType0', None)
    text_message = request.values.get('Body', '').strip()
    sender_number = request.values.get('From', 'Unknown')

    logger.info(f"Received message from {sender_number}")

    try:
        # Create or get session following ADK pattern
        session_id = f"twilio_{sender_number.replace(':', '_').replace('+', '')}"
        user_id = sender_number
        
        # Get or create session (use asyncio.run for sync Flask context)
        async def get_or_create_session():
            try:
                session = await session_service.get_session(
                    app_name="multimodal_assessment_app",
                    user_id=user_id, 
                    session_id=session_id
                )
                if not session:
                    session = await session_service.create_session(
                        app_name="multimodal_assessment_app",
                        user_id=user_id,
                        session_id=session_id,
                        state={"conversation_history": []}
                    )
                return session
            except Exception as e:
                logger.error(f"Session management error: {e}")
                # Create new session as fallback
                return await session_service.create_session(
                    app_name="multimodal_assessment_app",
                    user_id=user_id,
                    session_id=session_id,
                    state={"conversation_history": []}
                )
        
        session = asyncio.run(get_or_create_session())
        
        # Store the content in session state so the agent can access it
        if media_url and media_type:
            # For audio, we need to download and encode it for ADK
            logger.info(f"Processing audio via ADK runner: {media_url} (type: {media_type})")
            
            # Download audio
            audio_response = requests.get(
                media_url,
                auth=(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
            )
            audio_response.raise_for_status()
            audio_bytes = audio_response.content
            
            # Encode for ADK
            audio_data_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Create ADK content with audio
            content = types.Content(
                role='user',
                parts=[
                    types.Part(
                        inline_data=types.Blob(
                            mime_type=media_type,
                            data=audio_data_b64
                        )
                    )
                ]
            )
            
            # Store content in session state for the agent to access
            session.state["input_content"] = content
            session.state["input_type"] = "audio"
            logger.info(f"Stored audio content in session state: {type(content)}")
            
        elif text_message:
            logger.info(f"Processing text via ADK runner: {text_message[:50]}...")
            # Create ADK content with text
            content = types.Content(
                role='user',
                parts=[types.Part(text=text_message)]
            )
            
            # Store content in session state for the agent to access
            session.state["input_content"] = content
            session.state["input_type"] = "text"
            logger.info(f"Stored text content in session state: {type(content)}")
        else:
            # Default message
            result = "Hello! I'm your Google Ask Agent. Send me a voice note for audio assessment or text message for general assistance."
            twilio_response.message(result)
            logger.info("Sent default message")
            return str(twilio_response)
        
        # Run the agent using ADK Runner pattern (wrap in asyncio.run)
        logger.info("Running agent via ADK Runner...")
        
        async def run_agent_and_collect_response():
            events = runner.run_async(user_id=user_id, session_id=session_id, new_message=content)
            
            result = "Processing complete."
            async for event in events:
                # Check if this is a final response - use the correct ADK pattern
                if hasattr(event, 'content') and event.content and hasattr(event.content, 'parts') and event.content.parts:
                    logger.info(f"Response from [{event.author}]: {event.content.parts[0].text[:100]}...")
                    result = event.content.parts[0].text
                    # For now, take the first response as final
                    break
            return result
        
        result = asyncio.run(run_agent_and_collect_response())
        
        # Truncate message if too long for WhatsApp
        if len(result) > 1600:
            result = result[:1597] + "..."
            logger.warning(f"Message truncated to {len(result)} characters for WhatsApp limit")
        
        # Add the result to Twilio response
        twilio_response.message(result)
        logger.info(f"Added ADK response to Twilio: {len(result)} characters")

    except Exception as e:
        logger.error(f"Error in webhook: {e}")
        twilio_response.message("I encountered an error processing your request. Please try again later.")

    # Log the final TwiML response
    twiml_response = str(twilio_response)
    logger.info(f"Returning TwiML response: {twiml_response}")
    return twiml_response

@app.route("/health", methods=['GET'])
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "google_ask_agent_adk", 
        "version": "1.0.0",
        "agent_type": "ADK Custom Agent",
        "agent_name": multimodal_agent.name
    }, 200

@app.route("/", methods=['GET'])
def home():
    """Service information endpoint"""
    return {
        "service": "Google Ask Agent - ADK Multimodal AI Assistant",
        "description": "ADK Custom Agent that handles both text and audio inputs using Google's Gemini AI",
        "architecture": {
            "framework": "Google ADK (Agent Development Kit)",
            "pattern": "Custom Agent inheriting from BaseAgent",
            "core_method": "_run_async_impl",
            "wrapped_client": "genai.Client",
            "documentation": "https://google.github.io/adk-docs/agents/custom-agents/"
        },
        "capabilities": [
            "Audio assessment and transcription",
            "Text-based conversation and guidance", 
            "Skills evaluation and career advice",
            "Communication skills analysis",
            "ADK event handling and state management"
        ],
        "endpoints": {
            "/twilio-webhook": "POST - Webhook for WhatsApp/SMS messages",
            "/health": "GET - Health check",
            "/": "GET - Service information"
        },
        "supported_audio_formats": ["WAV", "MP3", "AIFF", "AAC", "OGG", "FLAC"],
        "adk_features": [
            "Custom agent orchestration logic",
            "Asynchronous event streaming", 
            "Session state management",
            "Proper ADK event yielding",
            "genai.Client integration"
        ],
        "version": "1.0.0"
    }, 200

@app.route("/agent-info", methods=['GET'])
def agent_info():
    """ADK agent-specific information endpoint"""
    return {
        "agent_name": multimodal_agent.name,
        "agent_type": "ADK Custom Agent",
        "base_class": "google.adk.agents.BaseAgent",
        "implementation": "_run_async_impl",
        "client": "genai.Client wrapped within ADK framework",
        "processing_capabilities": {
            "audio": "Via _process_audio_input method",
            "text": "Via _process_text_input method",
            "multimodal": "Automatic routing in _run_async_impl"
        },
        "state_management": "ctx.session.state (ADK standard)",
        "event_handling": "AsyncGenerator[Event, None]",
        "helper_methods": [
            "process_twilio_audio()",
            "process_text_message()"
        ]
    }, 200

if __name__ == '__main__':
    logger.info("🚀 Starting Google Ask Agent - ADK Multimodal AI Assistant")
    logger.info("🏗️  Using ADK Custom Agent architecture")
    logger.info("📱 Ready to handle both text and audio inputs via WhatsApp")
    logger.info("🎯 Powered by Google ADK + Gemini AI")
    logger.info(f"🤖 Agent: {multimodal_agent.name}")
    
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)), 
        debug=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    )