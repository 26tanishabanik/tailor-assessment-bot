import asyncio
import json
import logging
import os
from typing import Dict, Any
import tempfile

from agents.master_agent import MasterAgent
from agents.sub_agents.stitching_assessor_agent import StitchingAssessorAgent
from google.adk.agents import Agent
from google.adk.agents import LlmAgent
from google.genai import types # Import types for Blob

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Application:
    """
    The main host application that orchestrates the agent workflow.
    This application, not an agent, controls the flow of execution.
    """

    def __init__(self):
        self.master_agent = MasterAgent()
        # The host application maintains a library of available sub-agents.
        self.sub_agent_library = {
            "StitchingAssessorAgent": StitchingAssessorAgent()
        }
        logger.info("Host Application initialized.")

    async def execute_sub_agent_instruction(self, instruction: Dict, image_data: bytes = None) -> Dict:
        """Looks up and executes a sub-agent based on the Master Agent's plan."""
        agent_name = instruction.get('agent_name')
        task_context = instruction.get('task_context', {})
        
        if agent_name not in self.sub_agent_library:
            logger.error(f"Sub-agent '{agent_name}' not found in library.")
            return {"status": "error", "message": "Sub-agent not found."}
            
        sub_agent = self.sub_agent_library[agent_name]
        
        try:
            # The stitching assessor agent specifically expects image_data
            # If the initial input was audio, and the MasterAgent planned a stitching assessment,
            # you would need to ensure `image_data` is available here (e.g., from a previous image upload)
            # or modify the stitching_assessor_agent to handle non-image inputs if applicable.
            if image_data:
                return await sub_agent.execute(image_data, task_context)
            else:
                # Fallback to test image if no image_data is provided for a skill that requires it
                print(f"\n📸 Please provide an image for the '{task_context.get('skill_to_assess', 'Unknown')}' assessment.")
                print(f"   (Using default test image...)")
                with open("test_stitch.jpg", "rb") as f:
                    image_bytes = f.read()
                return await sub_agent.execute(image_bytes, task_context)
        except FileNotFoundError as e:
            error_msg = f"Image file not found: {e}"
            return {"status": "error", "message": error_msg}

    async def run_full_assessment(self, user_query: str = None, image_data: bytes = None, audio_data: bytes = None):
        """
        Runs the full, host-driven, two-step assessment process.
        """
        # STEP 1: Get an assessment plan from the Master Agent.
        # Pass all available modalities to the Master Agent
        plan = await self.master_agent.get_plan(user_query=user_query, image_data=image_data, audio_data=audio_data)
        print(f"\n🎯 Assessment Agent: {plan.get('response_to_user', '...')}")
        
        instructions = plan.get('sub_agent_instructions')
        if not instructions:
            return

        # STEP 2: The host executes the plan.
        sub_agent_results = []
        for instruction in instructions:
            skill_name = instruction.get('task_context', {}).get('skill_to_assess', 'Unknown')
            print(f"\n🔍 Executing {skill_name} Assessment...")
            # Pass image_data to sub-agent if available and required
            result = await self.execute_sub_agent_instruction(instruction, image_data=image_data)
            sub_agent_results.append({
                "agent_name": instruction.get('agent_name'),
                "skill_assessed": instruction.get('task_context', {}).get('skill_to_assess'),
                "result": result
            })
        
        # STEP 3: The host goes back to the Master Agent with the results for a final verdict.
        target_role = instructions[0].get('task_context', {}).get('role')
        if not target_role:
             print(f"\n❌ System Error: Could not determine role from assessment plan.")
             return
        
        print(f"\n🤔 Analyzing results for {target_role} position...")
        final_verdict = await self.master_agent.get_final_verdict(target_role, sub_agent_results)
        
        # STEP 4: Deliver the final, judged decision to the user.
        decision = final_verdict.get('final_decision_data', {}).get('decision', 'UNKNOWN')
        decision_emoji = '✅' if decision == 'PASS' else '❌' if decision == 'FAIL' else '⏳'
        print(f"\n{decision_emoji} Final Decision: {final_verdict.get('response_to_user', '...')}")


async def main():
    app = Application()
    print("\n🎯 Welcome to the Job Skill Assessment System!")
    print("=" * 50)
    print("Currently available role: Tailor")
    print("Usage examples:")
    print("  'I want to apply for Tailor position' - Will ask for image")
    print("  'Here is my work [test_stitch.jpg]' - Provide image with query")
    print("Type 'exit' or 'quit' to end the session")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n💼 User: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\n👋 Thank you for using the Job Assessment System. Goodbye!")
                break
            
            if not user_input:
                print("Please enter your message or type 'exit' to quit.")
                continue
            
            # This part is for CLI testing, in WhatsApp, the media_url handles this
            image_path = None
            audio_path = None

            # Check for image path in CLI input
            if '[' in user_input and ']' in user_input:
                start = user_input.find('[')
                end = user_input.find(']', start)
                if start != -1 and end != -1:
                    media_path = user_input[start+1:end].strip()
                    if media_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.avif')):
                        image_path = media_path
                    elif media_path.lower().endswith(('.mp3', '.wav', '.ogg', '.flac')):
                        audio_path = media_path
                    user_input = (user_input[:start] + user_input[end+1:]).strip()
            
            image_data = None
            audio_data = None

            if image_path:
                try:
                    with open(image_path, 'rb') as f:
                        image_data = f.read()
                    print(f"Loaded image from: {image_path}")
                except FileNotFoundError:
                    print(f"❌ Error: Image file not found at '{image_path}'.")
                    continue
            
            if audio_path:
                try:
                    # For CLI testing, convert OGG to WAV if needed, or ensure it's WAV
                    if audio_path.lower().endswith('.ogg'):
                        # Simulate conversion for CLI input
                        from pydub import AudioSegment
                        from pydub.utils import which
                        if not which("ffmpeg"):
                            print("ffmpeg not found. Cannot convert OGG for CLI testing.")
                            continue
                        ogg_audio = AudioSegment.from_ogg(audio_path)
                        wav_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                        ogg_audio.set_frame_rate(16000).set_channels(1).export(wav_temp_file.name, format="wav")
                        wav_temp_file.close()
                        with open(wav_temp_file.name, 'rb') as f:
                            audio_data = f.read()
                        os.unlink(wav_temp_file.name) # Clean up temp wav
                        print(f"Converted and loaded audio from: {audio_path}")
                    else: # Assume it's already WAV or compatible
                        with open(audio_path, 'rb') as f:
                            audio_data = f.read()
                        print(f"Loaded audio from: {audio_path}")
                except FileNotFoundError:
                    print(f"❌ Error: Audio file not found at '{audio_path}'.")
                    continue
                except Exception as e:
                    print(f"❌ Error processing audio file '{audio_path}': {e}")
                    continue

            print(f"\n🤖 Processing your request...")
            await app.run_full_assessment(user_query=user_input, image_data=image_data, audio_data=audio_data)
            
            print("\n" + "-" * 50)
            print("Ready for your next interaction!")
            
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try again or type 'exit' to quit.")

if __name__ == "__main__":
    asyncio.run(main())