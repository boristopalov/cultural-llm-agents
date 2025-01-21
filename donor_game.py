from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from profiles import CulturalProfile
import random
import re
import logging
import asyncio
from collections import deque
from pathlib import Path
from litellm import acompletion
from constants import (
    SYSTEM_PROMPT, STRATEGY_PROMPT_TEMPLATE, DONATION_PROMPT_TEMPLATE,
    MEMORY_SIZE, INITIAL_BALANCE, DONATION_MULTIPLIER, ROUNDS_PER_GENERATION
)
import time

# Configure root logger for stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@dataclass
class DonationDecision:
    """Represents a donation decision between two agents."""
    donor_id: str
    recipient_id: str
    amount: float
    success: bool
    punishment_amount: float = 0.0  # Amount spent on punishment
    error: Optional[str] = None

class Memory:
    """Simple memory implementation with fixed size."""
    def __init__(self, max_size: int = MEMORY_SIZE):
        self.messages = deque(maxlen=max_size)
        self.chat_history = deque(maxlen=max_size)  # Store chat messages for LLM context
    
    def store(self, message: str, chat_message: Optional[Dict] = None):
        """Store both a human-readable message and the chat message for LLM."""
        self.messages.append(message)
        if chat_message:
            # Ensure we're storing the message in the correct format
            if isinstance(chat_message.get('content'), list):
                # Extract text content if it's a list format
                content = chat_message['content'][0]['text'] if chat_message['content'] else ""
                chat_message = {"role": chat_message["role"], "content": content}
            self.chat_history.append(chat_message)
    
    def get_all_messages(self) -> List[str]:
        return list(self.messages)
    
    def get_chat_history(self) -> List[Dict]:
        return list(self.chat_history)

class DonorGameAgent:
    """An agent that participates in the donor game with cultural values."""
    
    def __init__(self, cultural_profile: CulturalProfile, agent_id: str, model: str = "anthropic/claude-3-sonnet-20240229"):
        self.cultural_profile = cultural_profile
        self.id = agent_id
        self.strategy = ""
        self.memory = Memory()
        self.model = model
        self.logger = logging.getLogger(f"agent.{agent_id}")
        self.logger.info(f"Initialized with cultural profile: {cultural_profile}")
    
    def _format_response_content(self, response) -> str:
        """Extract text content from different response formats."""
        if hasattr(response.choices[0], 'message'):
            content = response.choices[0].message.content
            if isinstance(content, list):
                return content[0]['text']
            return content
        return response.choices[0].text
    
    async def generate_strategy(self, generation: int, previous_gen_advice: str = "") -> None:
        """Generates a strategy for the agent using its cultural profile."""
        self.logger.info(f"Generating strategy for generation {generation}")
        
        if generation == 1:
            context = "Based on the description of the game, create a strategy that you will follow in the game."
        else:
            context = f"How would you approach the game?\nHere is the advice of the best-performing 50% of the previous generation, along with their final scores:\n{previous_gen_advice}\nModify this advice to create your own strategy."
        
        prompt = STRATEGY_PROMPT_TEMPLATE.format(
            name=self.id,
            context=context
        )
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self.memory.get_chat_history(),
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await acompletion(model=self.model, messages=messages)
            response_content = self._format_response_content(response)
            self.logger.info(f"Strategy generation response:\n{response_content}")
            
            strategy = self._extract_strategy(response_content)
            
            if not strategy:
                self.logger.warning("Strategy not found in initial response, retrying...")
                retry_prompt = f"""Your previous response did not include the required format. Here was your response:

{response_content}

Please reformulate your strategy so that it starts with exactly "My strategy will be". For example: "My strategy will be to donate 50% initially and adjust based on reciprocity."
"""
                messages.append({"role": "assistant", "content": response_content})
                messages.append({"role": "user", "content": retry_prompt})
                
                response = await acompletion(model=self.model, messages=messages)
                response_content = self._format_response_content(response)
                self.logger.info(f"Retry strategy generation response:\n{response_content}")
                strategy = self._extract_strategy(response_content)
            
            self.strategy = strategy or "My strategy will be to start with moderate donations and adjust based on recipient's history"
            self.logger.info(f"Final strategy: {self.strategy}")
            
            # Store the strategy discussion in memory
            self.memory.store(
                f"Generated strategy: {self.strategy}",
                {"role": "assistant", "content": response_content}
            )
            
        except Exception as e:
            self.logger.error(f"Error generating strategy: {e}")
            self.strategy = "My strategy will be to start with moderate donations and adjust based on recipient's history"
    
    async def make_donation_decision(self, generation: int, round: int, recipient_id: str, 
                                   recipient_resources: float, recipient_history: str, 
                                   donor_resources: float) -> DonationDecision:
        """Decides how much to donate and/or punish based on the current situation."""
        self.logger.info(f"Making donation decision in generation {generation}, round {round}")
        self.logger.info(f"Context: I have {donor_resources:.2f} resources, {recipient_id} has {recipient_resources:.2f}")
        
        if recipient_history:
            self.logger.info(f"Recipient history:\n{recipient_history}")
        
        prompt = DONATION_PROMPT_TEMPLATE.format(
            name=self.id,
            strategy=self.strategy,
            generation=generation,
            round=round,
            recipient=recipient_id,
            recipient_resources=recipient_resources,
            recipient_history=recipient_history,
            donor_resources=donor_resources
        )
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self.memory.get_chat_history()[-5:],  # Include last 5 interactions for context
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await acompletion(model=self.model, messages=messages)
            response_content = self._format_response_content(response)
            self.logger.info(f"Donation decision response:\n{response_content}")
            
            donation, punishment = self._parse_decision_response(response_content)
            total_cost = donation + punishment
            
            if total_cost > donor_resources:
                self.logger.warning(f"Attempted to spend {total_cost:.2f} but only have {donor_resources:.2f}")
                # Scale down both donation and punishment proportionally
                scale = donor_resources / total_cost
                donation *= scale
                punishment *= scale
            
            # Store the decision in memory
            self.memory.store(
                f"Made decision: donation={donation:.2f}, punishment={punishment:.2f}",
                {"role": "assistant", "content": response_content}
            )
            
            self.logger.info(f"Final decision: donation={donation:.2f}, punishment={punishment:.2f} to {recipient_id}")
            return DonationDecision(
                donor_id=self.id,
                recipient_id=recipient_id,
                amount=donation,
                punishment_amount=punishment,
                success=True
            )
            
        except Exception as e:
            error_msg = f"Error making donation decision: {e}"
            self.logger.error(error_msg)
            return DonationDecision(
                donor_id=self.id,
                recipient_id=recipient_id,
                amount=0.0,
                punishment_amount=0.0,
                success=False,
                error=error_msg
            )
    
    def _extract_strategy(self, response: str) -> Optional[str]:
        """Extracts the strategy from the LLM response."""
        for line in response.split('\n'):
            if line.lower().startswith('my strategy will be'):
                return line.strip()
        return None
    
    def _parse_decision_response(self, response: str) -> Tuple[float, float]:
        """Extracts the donation and punishment amounts from the LLM response."""
        donation_match = re.search(r'DONATION:\s*(\d*\.?\d+)', response)
        punishment_match = re.search(r'PUNISHMENT:\s*(\d*\.?\d+)', response)
        
        if not donation_match or not punishment_match:
            self.logger.warning(f"Could not find both donation and punishment in response: {response}")
            # Default to 0 for any missing values
            donation = float(donation_match.group(1)) if donation_match else 0.0
            punishment = float(punishment_match.group(1)) if punishment_match else 0.0
            return donation, punishment
        
        return float(donation_match.group(1)), float(punishment_match.group(1))

@dataclass
class DonorGameState:
    """Represents the current state of the donor game."""
    round: int = 0
    total_rounds: int = 0
    agent_resources: Dict[str, float] = None
    successful_donations: int = 0
    failed_donations: int = 0

    def __post_init__(self):
        if self.agent_resources is None:
            self.agent_resources = {}

class DonorGameEnvironment:
    """Manages the donor game environment and interactions between agents."""
    
    def __init__(self, rounds_per_gen: int = ROUNDS_PER_GENERATION, 
                 donation_mult: float = DONATION_MULTIPLIER, 
                 initial_balance: float = INITIAL_BALANCE):
        self.agents: List[DonorGameAgent] = []
        self.state = DonorGameState()
        self.rounds_per_gen = rounds_per_gen
        self.donation_mult = donation_mult
        self.initial_balance = initial_balance
        self.logger = logging.getLogger("environment")
        self.donation_callback = None  # Callback for donation events
    
    def set_donation_callback(self, callback) -> None:
        """Set callback function to be called after each donation."""
        self.donation_callback = callback
    
    def add_agent(self, agent: DonorGameAgent, initial_resources: Optional[float] = None) -> None:
        """Adds an agent to the environment with optional initial resources."""
        self.agents.append(agent)
        if initial_resources is not None:
            self.state.agent_resources[agent.id] = initial_resources
        else:
            self.state.agent_resources[agent.id] = self.initial_balance
    
    def remove_agent(self, agent: DonorGameAgent) -> None:
        """Removes an agent from the environment."""
        self.agents.remove(agent)
        del self.state.agent_resources[agent.id]
    
    def reset(self, keep_resources: bool = False, survivor_ids: Optional[List[str]] = None) -> None:
        """
        Resets the environment for a new generation.
        
        Args:
            keep_resources: If True, saves current agent resources before reset
            survivor_ids: List of agent IDs whose resources should be kept
        """
        saved_resources = {}
        if keep_resources and survivor_ids:
            # Only keep resources for surviving agents
            saved_resources = {
                agent_id: self.state.agent_resources[agent_id]
                for agent_id in survivor_ids
                if agent_id in self.state.agent_resources
            }
        
        self.state = DonorGameState()
        
        # Clear agents but keep track of resources if needed
        self.agents = []
        if keep_resources and saved_resources:
            self.state.agent_resources = saved_resources
    
    async def step(self) -> None:
        """Runs one round of the donor game."""
        if len(self.agents) % 2 != 0:
            raise ValueError("Need even number of agents")
        
        self.logger.info(f"\n=== Round {self.state.round} ===")
        self.logger.info("Current resources:")
        for agent_id, resources in self.state.agent_resources.items():
            self.logger.info(f"  {agent_id}: {resources:.2f}")
        
        # Shuffle agents for random pairing
        agents = self.agents.copy()
        random.shuffle(agents)
        
        # Create pairs and gather their decisions concurrently
        pairs = [(agents[i], agents[i+1]) for i in range(0, len(agents), 2)]
        self.logger.info("\nStarting parallel donation decisions...")
        start_time = time.time()
        
        decisions = await asyncio.gather(
            *[self._get_donation_decision(donor, recipient) for donor, recipient in pairs]
        )
        
        end_time = time.time()
        self.logger.info(f"All donation decisions completed in {end_time - start_time:.2f} seconds")
        
        # Process all donations
        for decision in decisions:
            if decision.success:
                self._process_donation(
                    next(a for a in self.agents if a.id == decision.donor_id),
                    next(a for a in self.agents if a.id == decision.recipient_id),
                    decision.amount,
                    decision.punishment_amount
                )
                if decision.amount > 0:
                    self.state.successful_donations += 1
                else:
                    self.state.failed_donations += 1
            else:
                self.logger.error(f"Failed donation from {decision.donor_id} to {decision.recipient_id}: {decision.error}")
                self.state.failed_donations += 1
        
        # Print round summary
        self._log_round_summary()
        
        # Update round counters
        self.state.round += 1
        self.state.total_rounds += 1
        if self.state.round >= self.rounds_per_gen:
            self.state.round = 0
    
    async def _get_donation_decision(self, donor: DonorGameAgent, recipient: DonorGameAgent) -> DonationDecision:
        """Gets a donation decision from a donor agent."""
        start_time = time.time()
        self.logger.info(f"Starting decision: {donor.id} -> {recipient.id}")
        
        recipient_history = self._get_recent_history(recipient.id)
        
        decision = await donor.make_donation_decision(
            generation=self.state.total_rounds // self.rounds_per_gen + 1,
            round=self.state.round,
            recipient_id=recipient.id,
            recipient_resources=self.state.agent_resources[recipient.id],
            recipient_history=recipient_history,
            donor_resources=self.state.agent_resources[donor.id]
        )
        
        end_time = time.time()
        self.logger.info(f"Decision completed: {donor.id} -> {recipient.id} in {end_time - start_time:.2f} seconds")
        
        return decision
    
    def _process_donation(self, donor: DonorGameAgent, recipient: DonorGameAgent, amount: float, punishment_amount: float) -> None:
        """Processes a donation and punishment between two agents."""
        donor_resources = self.state.agent_resources[donor.id]
        pct_donation = amount / donor_resources if donor_resources > 0 else 0
        pct_punishment = punishment_amount / donor_resources if donor_resources > 0 else 0
        
        # Process donation
        self.state.agent_resources[donor.id] -= amount
        multiplied_amount = amount * self.donation_mult
        self.state.agent_resources[recipient.id] += multiplied_amount
        
        # Process punishment
        if punishment_amount > 0:
            self.state.agent_resources[donor.id] -= punishment_amount
            punishment_effect = punishment_amount * 2  # Punishment multiplier
            self.state.agent_resources[recipient.id] = max(0, self.state.agent_resources[recipient.id] - punishment_effect)
        
        # Log the interaction
        if amount > 0:
            self.logger.info(f"  {donor.id} donated {amount:.2f} ({pct_donation:.1%}) to {recipient.id}")
            self.logger.info(f"  {recipient.id} received {multiplied_amount:.2f} (x{self.donation_mult})")
        
        if punishment_amount > 0:
            self.logger.info(f"  {donor.id} spent {punishment_amount:.2f} ({pct_punishment:.1%}) to punish {recipient.id}")
            self.logger.info(f"  {recipient.id} lost {punishment_effect:.2f} resources")
        
        # Notify callback if set
        if self.donation_callback:
            self.donation_callback(donor.id, recipient.id, amount, punishment_amount)
        
        # Update memories
        interaction = []
        if amount > 0:
            interaction.append(f"donated {pct_donation:.2%} ({amount:.2f})")
        if punishment_amount > 0:
            interaction.append(f"spent {pct_punishment:.2%} ({punishment_amount:.2f}) on punishment")
        
        donor_memory = f"Round: I {' and '.join(interaction)} with {recipient.id}"
        
        interaction = []
        if amount > 0:
            interaction.append(f"received {pct_donation:.2%} ({multiplied_amount:.2f})")
        if punishment_amount > 0:
            interaction.append(f"was punished and lost {punishment_effect:.2f}")
        
        recipient_memory = f"Round: I {' and '.join(interaction)} from {donor.id}"
        
        donor.memory.store(donor_memory)
        recipient.memory.store(recipient_memory)
    
    def _get_recent_history(self, agent_id: str) -> str:
        """Gets the recent interaction history for an agent."""
        for agent in self.agents:
            if agent.id == agent_id:
                memories = agent.memory.get_all_messages()[-3:]  # Get last 3 memories
                if not memories:
                    return "This is the first round, so there is no history of previous interactions."
                return "\n".join(memories)
        return ""
    
    def _log_round_summary(self) -> None:
        """Log summary statistics for the current round."""
        resources = list(self.state.agent_resources.values())
        total = sum(resources)
        avg = total / len(resources)
        min_r = min(resources)
        max_r = max(resources)
        
        self.logger.info("\nRound Summary:")
        self.logger.info(f"  Total Resources: {total:.2f}")
        self.logger.info(f"  Average Resources: {avg:.2f}")
        self.logger.info(f"  Resource Range: {min_r:.2f} - {max_r:.2f}")
        self.logger.info(f"  Successful Donations: {self.state.successful_donations}")
        self.logger.info(f"  Failed Donations: {self.state.failed_donations}")
        success_rate = (self.state.successful_donations / 
                       (self.state.successful_donations + self.state.failed_donations) * 100
                       if self.state.successful_donations + self.state.failed_donations > 0 else 0)
        self.logger.info(f"  Success Rate: {success_rate:.1f}%")
        self.logger.info("=" * 40)
    
    def get_top_agents(self, n: int) -> List[str]:
        """Returns the IDs of the top n performing agents."""
        sorted_agents = sorted(
            self.state.agent_resources.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [agent_id for agent_id, _ in sorted_agents[:n]] 
