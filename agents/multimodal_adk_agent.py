import os
import base64
import requests
import json
import logging
from typing import AsyncGenerator, Dict, Any, Optional
from pydantic import Field, model_validator, PrivateAttr
from google.adk.agents import BaseAgent, Agent
from google.genai.types import Content, Part
from google.adk.events import Event
from google.adk.agents.invocation_context import InvocationContext
from google import genai
from sub_agents.stitching_assessor_agent import stitching_assessor
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultimodalADKAgent(BaseAgent):
    """
    A custom ADK agent that handles both text and audio inputs following Google ADK pattern.
    
    Features:
    1. Inherits from BaseAgent (Pydantic model)
    2. Implements _run_async_impl for custom orchestration logic
    3. Uses Runner and InMemorySessionService for session management
    4. Integrates with stitching sub-agent for specialized assessments
    5. Loads knowledge base like master agent`
    """
    
    # Pydantic field declarations for sub-agents (like in the example)
    stitching_assessor_agent: Agent = Field(default=stitching_assessor)
    
    # Private attributes for internal state
    _client: Optional[genai.Client] = PrivateAttr(default=None)
    _master_prompt_template: Optional[str] = PrivateAttr(default=None)
    _competency_map: Optional[Dict] = PrivateAttr(default=None)
    _sub_agent_library: Optional[Dict] = PrivateAttr(default=None)
    _instruction: Optional[str] = PrivateAttr(default=None)
    _model_name: Optional[str] = PrivateAttr(default=None)
    
    # Model config to allow arbitrary types
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self, name: str = "MultimodalAgent", prompts_dir: str = "agent_prompts"):
        """
        Initialize the MultimodalADKAgent following ADK pattern.
        
        Args:
            name: The name of the agent
            prompts_dir: Directory containing agent prompts and knowledge base
        """
        # Load knowledge base before calling super().__init__
        master_prompt_template, competency_map, sub_agent_library = self._load_knowledge(prompts_dir)
        
        # Create instruction like master agent
        instruction = master_prompt_template + "\n\n--- KNOWLEDGE BASE ---\n" + \
                     "\nCompetency Map:\n" + json.dumps(competency_map, indent=2) + \
                     "\n\nSub-Agent Library:\n" + json.dumps(sub_agent_library, indent=2) + \
                     "\n\nYou can delegate to specialized sub-agents based on the role assessment needed. " + \
                     "Use your sub-agents for technical skill evaluation. You also handle audio and text inputs directly."
        
        # Initialize with proper ADK pattern - BaseAgent only accepts specific fields
        super().__init__(
            name=name,
            stitching_assessor_agent=stitching_assessor,
            sub_agents=[stitching_assessor]  # List of sub-agents
        )
        
        # Store the instruction and model info in private attributes for use in _run_async_impl
        self._instruction = instruction
        self._model_name = "gemini-2.0-flash-thinking-exp"
        
        # Store knowledge in private attributes
        self._master_prompt_template = master_prompt_template
        self._competency_map = competency_map
        self._sub_agent_library = sub_agent_library
    
    def _load_knowledge(self, prompts_dir: str) -> tuple:
        """Load knowledge base files like master agent does"""
        try:
            # Try to load from audio_support directory first, then fallback
            base_paths = [
                prompts_dir,
                os.path.join("..", prompts_dir),
                os.path.join("..", "..", prompts_dir)
            ]
            
            for base_path in base_paths:
                try:
                    with open(os.path.join(base_path, 'master_agent_prompt.md'), 'r') as f:
                        master_prompt_template = f.read()
                    with open(os.path.join(base_path, 'competency_map.json'), 'r') as f:
                        competency_map = json.load(f)
                    with open(os.path.join(base_path, 'sub_agent_library.json'), 'r') as f:
                        sub_agent_library = json.load(f)
                    
                    logger.info(f"Loaded knowledge base from: {base_path}")
                    return master_prompt_template, competency_map, sub_agent_library
                except FileNotFoundError:
                    continue
                    
            raise FileNotFoundError(f"Could not find knowledge base files in any of: {base_paths}")
                    
        except Exception as e:
            logger.warning(f"Could not load knowledge base: {e}")
    
    
    @model_validator(mode='after')
    def initialize_client(self):
        """Initialize the genai client after Pydantic model validation"""
        # Try GOOGLE_API_KEY first (preferred), then fall back to GEMINI_API_KEY
        api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found in environment variables.")
        
        self._client = genai.Client(api_key=api_key)
        logger.info(f"Initialized {self.name} with genai.Client and knowledge base")
        return self
    
    @property
    def client(self) -> genai.Client:
        """Property to access the genai client"""
        if self._client is None:
            raise RuntimeError("Client not initialized. Call initialize_client first.")
        return self._client
    
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Core ADK custom agent implementation following the _run_async_impl pattern.
        
        This method orchestrates multimodal processing:
        1. Analyzes the input content to determine if it's text or audio
        2. Uses master agent knowledge base for assessment logic
        3. Delegates to sub-agents when specialized skills assessment is needed
        4. Manages state through ctx.session.state
        5. Yields proper ADK events
        """
        logger.info(f"[{self.name}] Starting multimodal processing with ADK pattern")
        
        try:
            # Debug the context structure to understand how to access input
            logger.info(f"[{self.name}] Context attributes: {dir(ctx)}")
            logger.info(f"[{self.name}] Context type: {type(ctx)}")
            
            # Try to get content from different possible sources
            content = None
            input_type = None
            
            # Method 1: Try session state
            if hasattr(ctx, 'session') and hasattr(ctx.session, 'state'):
                content = ctx.session.state.get('input_content')
                input_type = ctx.session.state.get('input_type')
                logger.info(f"[{self.name}] Session state keys: {list(ctx.session.state.keys())}")
                logger.info(f"[{self.name}] Found content in session state: {input_type}, content type: {type(content) if content else 'None'}")
            
            # Method 2: Try context attributes
            if not content:
                for attr_name in dir(ctx):
                    if not attr_name.startswith('_'):
                        attr_value = getattr(ctx, attr_name)
                        logger.info(f"[{self.name}] Context attr {attr_name}: {type(attr_value)}")
                        if hasattr(attr_value, 'parts') or hasattr(attr_value, 'text'):
                            content = attr_value
                            input_type = "context_attr"
                            logger.info(f"[{self.name}] Found content in context attr: {attr_name}")
                            break
            
            # Process the content using the master prompt
            if content:
                logger.info(f"[{self.name}] Processing {input_type} content")
                async for event in self._process_input_with_master_prompt(ctx, content):
                    yield event
            else:
                logger.info(f"[{self.name}] No content found, providing default response")
                response_text = (
                    "Hello! I'm your multimodal AI assistant for skills assessment. "
                    "I'm currently processing your request using the ADK framework. "
                    "Send me a voice note for audio assessment or text message describing your experience. "
                    "I can evaluate skills for blue-collar roles like Tailoring."
                )
                yield self._create_text_event(response_text)
                
        except Exception as e:
            logger.error(f"[{self.name}] Error in _run_async_impl: {e}")
            yield self._create_error_event(f"An error occurred: {str(e)}")
    
    async def _process_input_with_master_prompt(self, ctx: InvocationContext, content: Content) -> AsyncGenerator[Event, None]:
        """Process any input (text or audio) using the master prompt with placeholders"""
        logger.info(f"[{self.name}] Processing input with master prompt")
        
        try:
            # Prepare the master prompt with knowledge base
            master_prompt = f"""
{self._instruction}

KNOWLEDGE BASE:
{json.dumps(self._competency_map, indent=2)}

SUB-AGENT LIBRARY:
{json.dumps(self._sub_agent_library, indent=2)}

{{INPUT_PLACEHOLDER}}
"""
            
            # Prepare content parts for genai API
            content_parts = [master_prompt]
            input_type = ""
            
            # Check what type of input we have and prepare accordingly
            for part in content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    # Audio input
                    input_type = "audio"
                    ctx.session.state["input_type"] = "audio"
                    
                    # Replace placeholder with audio instruction
                    audio_instruction = "The user has provided an audio recording for assessment. Transcribe and evaluate it following your role as a master job assessor."
                    content_parts[0] = master_prompt.replace("{INPUT_PLACEHOLDER}", audio_instruction)
                    
                    # Add audio part
                    audio_part = {
                        "inline_data": {
                            "mime_type": part.inline_data.mime_type,
                            "data": part.inline_data.data
                        }
                    }
                    content_parts.append(audio_part)
                    break
                    
                elif hasattr(part, 'text') and part.text:
                    # Text input
                    input_type = "text"
                    ctx.session.state["input_type"] = "text"
                    
                    # Replace placeholder with text instruction
                    text_instruction = f'User message: "{part.text}"\n\nAnalyze this message and provide appropriate guidance following your role as a master job assessor.'
                    content_parts[0] = master_prompt.replace("{INPUT_PLACEHOLDER}", text_instruction)
                    
                    ctx.session.state["user_input"] = part.text
                    break
            
            if not input_type:
                yield self._create_error_event("No valid input found")
                return
            
            # Process with genai client using master prompt
            response = self.client.models.generate_content(
                model=self._model_name,
                contents=content_parts
            )
            
            # Store results in session state
            ctx.session.state[f"{input_type}_response"] = response.text
            ctx.session.state["assessment_complete"] = True
            
            # Extract clean response text (remove JSON formatting if present)
            response_text = response.text
            if response_text.startswith('```json'):
                # Extract the actual response from JSON format
                try:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    if json_start != -1 and json_end != -1:
                        json_str = response_text[json_start:json_end]
                        json_data = json.loads(json_str)
                        response_text = json_data.get('response_to_user', response_text)
                except Exception as e:
                    logger.warning(f"JSON parsing failed: {e}, using original text")
                    # If JSON parsing fails, use the original text
                    pass
            
            # Check if sub-agent delegation is needed
            if self._should_delegate_to_stitching_agent(response_text):
                logger.info(f"[{self.name}] Potential stitching assessment detected")
                final_response = f"{response_text}\n\nFor detailed stitching skill evaluation, please also share an image of your work."
            else:
                final_response = response_text
            
            yield self._create_text_event(final_response)
            
        except Exception as e:
            logger.error(f"[{self.name}] Error processing input: {e}")
            yield self._create_error_event(f"Error processing input: {str(e)}")
    
    def _should_delegate_to_stitching_agent(self, text: str) -> bool:
        """Determine if stitching sub-agent should be used"""
        stitching_keywords = ["stitch", "sewing", "tailor", "fabric", "thread", "needle", "seam"]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in stitching_keywords)
    
    def _has_audio_content(self, content: Content) -> bool:
        """Check if the content contains audio parts"""
        if not content or not content.parts:
            return False
        
        for part in content.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                mime_type = getattr(part.inline_data, 'mime_type', '')
                if mime_type.startswith('audio/'):
                    return True
        return False
    
    def _has_text_content(self, content: Content) -> bool:
        """Check if the content contains text parts"""
        if not content or not content.parts:
            return False
        
        for part in content.parts:
            if hasattr(part, 'text') and part.text:
                return True
        return False
    

    
    def _create_text_event(self, text: str) -> Event:
        """Helper method to create a text response event"""
        try:
            # Direct Content creation without from_parts
            content = Content(
                role='assistant',
                parts=[Part(text=text)]
            )
            return Event(
                content=content,
                author=self.name
            )
        except Exception as e:
            logger.error(f"Error creating text event: {e}")
            # Fallback to simpler event creation
            return Event(
                content=Content(parts=[Part(text=text)]),
                author=self.name
            )

    def _create_error_event(self, error_message: str) -> Event:
        """Helper method to create an error event"""
        try:
            # Direct Content creation without from_parts
            content = Content(
                role='assistant',
                parts=[Part(text=error_message)]
            )
            return Event(
                content=content,
                author=self.name
            )
        except Exception as e:
            logger.error(f"Error creating error event: {e}")
            # Fallback to simpler event creation
            return Event(
                content=Content(parts=[Part(text=error_message)]),
                author=self.name
        )

