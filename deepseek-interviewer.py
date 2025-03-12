import os
from typing import List, Dict, Any
import tiktoken
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class InterviewBot:
    def __init__(self, api_key: str):
        """Initialize the InterviewBot with Groq API key."""
        self.api_key = api_key
        self.client = Groq(api_key=self.api_key)
        self.model = "deepseek-r1-distill-llama-70b"
        self.conversation_history = []
        self.question_count = 0
        self.follow_up_count = 0
        self.current_topic = None
        self.max_follow_ups = 2
        self.max_questions = 10
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # OpenAI's encoding for comparison
        
        # Token usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        # Per-question token tracking
        self.per_question_tokens = []
        
        # Initial system prompt to guide the Groq model
        self.system_prompt = {
            "role": "system", 
            "content": """
            You are an interview AI. Your task is to conduct a thorough interview with the user.
            Follow these guidelines:
            1. Ask one question at a time
            2. Label each question with its number (e.g., "Question 1:")
            3. Keep your questions and responses under 200 tokens
            4. Adapt your next question based on the user's previous answer
            5. IMPORTANT: Limit follow-up questions on the same topic to a maximum of 2
            6. After 1-2 follow-ups, move to a completely new topic
            7. Conduct exactly 10 questions in total
            8. Be conversational but focused
            9. Track follow-up count for each topic and reset when changing topics
            
            Start by introducing yourself briefly and asking the first question.
            """
        }
        
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a string."""
        return len(self.tokenizer.encode(text))
    
    def format_user_instructions(self) -> str:
        """Provide token limit instructions to the user."""
        return "Please keep your response under 200 tokens (approximately 40-50 words)."
    
    def generate_next_question(self, user_response: str = None) -> str:
        """Generate the next interview question based on conversation history."""
        self.question_count += 1
        
        # Initialize token tracking for this question
        question_token_data = {
            "question_number": self.question_count,
            "prompt_tokens": 0,
            "user_tokens": 0,
            "question_tokens": 0,
            "total_input": 0,
            "total_output": 0
        }
        
        # Initialize messages with system prompt
        messages = [self.system_prompt]
        
        # For the first question, we don't have a user response yet
        if self.question_count == 1:
            instruction = f"Generate Question {self.question_count}:"
            messages.append({"role": "user", "content": instruction})
            self.follow_up_count = 0  # Initialize follow-up count
            
            # Track prompt tokens for the first question
            prompt_tokens = self.count_tokens(self.system_prompt["content"]) + self.count_tokens(instruction)
            question_token_data["prompt_tokens"] = prompt_tokens
            self.total_input_tokens += prompt_tokens
        else:
            # Add the user response to conversation history
            self.conversation_history.append({"role": "user", "content": user_response})
            
            # Track user input tokens
            user_tokens = self.count_tokens(user_response)
            self.total_input_tokens += user_tokens
            question_token_data["user_tokens"] = user_tokens
            
            # Add conversation history to messages
            for message in self.conversation_history:
                messages.append(message)
            
            # Create instruction for the next question
            instruction = ""
            
            # Determine if we should continue with follow-ups or change topic
            self.follow_up_count += 1
            if self.follow_up_count >= self.max_follow_ups:
                instruction += f"\nYou've asked {self.follow_up_count} follow-up questions on this topic. "
                instruction += f"For Question {self.question_count}, introduce a completely new topic unrelated to previous questions:"
                self.follow_up_count = 0  # Reset follow-up count for new topic
            else:
                instruction += f"\nThis is follow-up #{self.follow_up_count} on the current topic. "
                instruction += f"Generate Question {self.question_count} based on the conversation:"
            
            # Check if we need to specifically instruct to conclude the interview
            if self.question_count >= self.max_questions - 1:
                instruction += "\nThis should be one of the final questions. The next question will conclude the interview."
            elif self.question_count >= self.max_questions:
                instruction += "\nThis should be the final question to conclude the interview."
            
            messages.append({"role": "user", "content": instruction})
            
            # Calculate prompt tokens for this iteration
            prompt_tokens = self.count_tokens(instruction)
            # Add tokens for all messages in history
            for msg in self.conversation_history:
                prompt_tokens += self.count_tokens(msg["content"])
            # Add system prompt tokens
            prompt_tokens += self.count_tokens(self.system_prompt["content"])
            
            question_token_data["prompt_tokens"] = prompt_tokens
            self.total_input_tokens += prompt_tokens
        
        # Generate the AI response using Groq
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.6,
                max_completion_tokens=4096,
                top_p=0.95,
                stream=False,
                stop=None,
            )
            
            # Extract the generated question
            question_text = completion.choices[0].message.content
            
            # Track response output tokens
            response_tokens = self.count_tokens(question_text)
            self.total_output_tokens += response_tokens
            question_token_data["question_tokens"] = response_tokens
            
            # Ensure the question is within the token limit
            if response_tokens > 200:
                # Truncate if needed
                question_text = self.truncate_to_token_limit(question_text, 190) + "..."
                # Recalculate tokens after truncation
                self.total_output_tokens -= response_tokens
                response_tokens = self.count_tokens(question_text)
                self.total_output_tokens += response_tokens
                question_token_data["question_tokens"] = response_tokens
            
            # Add the model's response to conversation history
            self.conversation_history.append({"role": "assistant", "content": question_text})
            
            # Add token limit reminder for user
            full_response = (
                f"{question_text}\n\n"
                f"{self.format_user_instructions()}"
            )
            
            # Update token totals for this question
            question_token_data["total_input"] = question_token_data["prompt_tokens"] + question_token_data["user_tokens"]
            question_token_data["total_output"] = question_token_data["question_tokens"]
            
            # Store the token data for this question
            self.per_question_tokens.append(question_token_data)
            
            return full_response
            
        except Exception as e:
            error_msg = f"Error generating question: {str(e)}"
            print(error_msg)
            return f"Error: {error_msg}\n\nPlease try again."
    
    def truncate_to_token_limit(self, text: str, limit: int) -> str:
        """Truncate text to stay within token limit."""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= limit:
            return text
        
        return self.tokenizer.decode(tokens[:limit])
    
    def validate_user_response(self, response: str) -> str:
        """Validate user response is within token limits."""
        token_count = self.count_tokens(response)
        
        if token_count > 200:
            truncated_response = self.truncate_to_token_limit(response, 200) + "\n[Note: Your response was truncated to 200 tokens]"
            # Adjust token count after truncation
            token_count = 200  # Count only the tokens that will be used
            # Update the per-question token tracking for the current question
            if self.per_question_tokens and self.question_count <= len(self.per_question_tokens):
                self.per_question_tokens[self.question_count-1]["user_tokens"] = token_count
                self.per_question_tokens[self.question_count-1]["total_input"] = (
                    self.per_question_tokens[self.question_count-1]["prompt_tokens"] + token_count
                )
            return truncated_response
        
        # Update the per-question token tracking for the current question
        if self.per_question_tokens and self.question_count <= len(self.per_question_tokens):
            self.per_question_tokens[self.question_count-1]["user_tokens"] = token_count
            self.per_question_tokens[self.question_count-1]["total_input"] = (
                self.per_question_tokens[self.question_count-1]["prompt_tokens"] + token_count
            )
        
        # Track tokens for the full response
        self.total_input_tokens += token_count
        return response
    
    def is_interview_complete(self) -> bool:
        """Check if the interview should be concluded."""
        # Exactly 10 questions
        return self.question_count >= self.max_questions
        
    def get_token_usage_summary(self) -> dict:
        """Get the summary of token usage throughout the interview."""
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "questions_asked": self.question_count,
            "per_question_tokens": self.per_question_tokens
        }

def run_interview():
    """Run the interview bot application."""
    # Get API key from environment variable or input
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        api_key = input("Enter your Groq API key: ")
    
    # Initialize interview bot
    bot = InterviewBot(api_key)
    
    print("=" * 50)
    print("Welcome to the AI Interview Bot (Groq - deepseek-r1-distill-llama-70b)")
    print("=" * 50)
    print("Starting the interview. Please answer each question.")
    print("Your responses should be under 200 tokens (approximately 40-50 words).")
    print("=" * 50)
    
    start_time = time.time()
    
    # Start with the first question
    current_question = bot.generate_next_question()
    print(current_question)
    
    # Continue the interview until completion
    while not bot.is_interview_complete():
        # Get user response
        user_response = input("\nYour answer: ")
        print()
        
        # Validate and potentially truncate user response
        validated_response = bot.validate_user_response(user_response)
        if validated_response != user_response:
            print(validated_response)
        
        # Generate and display next question
        next_question = bot.generate_next_question(validated_response)
        print(next_question)
    
    # Calculate total interview time
    interview_duration = time.time() - start_time
    
    # Get token usage statistics
    token_usage = bot.get_token_usage_summary()
    
    print("\n" + "=" * 50)
    print("Interview complete! Thank you for your participation.")
    print("=" * 50)
    print("\nInterview Statistics:")
    print(f"Total questions asked: {token_usage['questions_asked']}")
    print(f"Interview duration: {interview_duration:.2f} seconds")
    
    # Display per-question token usage
    print("\nPer-Question Token Usage:")
    print("-" * 80)
    print(f"{'Q#':<4}{'Question Tokens':<18}{'User Tokens':<16}{'Prompt Tokens':<18}{'Total':<10}")
    print("-" * 80)
    
    for q_data in token_usage['per_question_tokens']:
        q_num = q_data['question_number']
        q_tokens = q_data['question_tokens']
        u_tokens = q_data['user_tokens']
        p_tokens = q_data['prompt_tokens']
        total = q_data['total_input'] + q_data['total_output']
        
        print(f"{q_num:<4}{q_tokens:<18}{u_tokens:<16}{p_tokens:<18}{total:<10}")
    
    print("-" * 80)
    
    # Display overall token usage
    print("\nToken Usage Summary:")
    print(f"Input tokens: {token_usage['input_tokens']}")
    print(f"Output tokens: {token_usage['output_tokens']}")
    print(f"Total tokens: {token_usage['total_tokens']}")
    
    # Estimate cost (based on approximate Groq pricing - adjust as needed)
    # Note: These are placeholder values - actual Groq pricing may differ
    input_cost = token_usage['input_tokens'] * 0.0000005  # $0.0005 per 1K input tokens (placeholder)
    output_cost = token_usage['output_tokens'] * 0.0000015  # $0.0015 per 1K output tokens (placeholder)
    total_cost = input_cost + output_cost
    
    print(f"\nEstimated Cost (USD - approximate):")
    print(f"Input cost: ${input_cost:.6f}")
    print(f"Output cost: ${output_cost:.6f}")
    print(f"Total cost: ${total_cost:.6f}")
    print("Note: Actual Groq pricing may differ from these estimates.")
    print("=" * 50)

if __name__ == "__main__":
    # Install required dependencies if needed
    try:
        import tiktoken
        import groq
    except ImportError:
        print("Installing required packages...")
        os.system("pip install tiktoken groq")
        
    run_interview()