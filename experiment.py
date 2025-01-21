from typing import List, Callable, Optional, Dict
import logging
import math
import time
from pathlib import Path
import csv
from dataclasses import dataclass
from donor_game import DonorGameAgent, DonorGameEnvironment
import asyncio

@dataclass
class GenerationStats:
    """Statistics for a single generation."""
    total_resources: float
    avg_resources: float
    std_dev: float
    resource_inequality: float
    successful_donations: int
    failed_donations: int
    success_rate: float

@dataclass
class AgentHistory:
    """Tracks an agent's history across generations."""
    agent_id: str
    original_generation: int
    generations_survived: int
    total_donations_made: int
    total_amount_donated: float
    total_amount_received: float
    total_punishments_given: int
    total_punishments_received: int
    total_amount_spent_on_punishment: float
    total_amount_lost_to_punishment: float
    strategies: List[str]
    resources_over_time: List[float]

class DonorGameExperiment:
    """Runs the donor game with generational evolution."""
    
    def __init__(
        self,
        env: DonorGameEnvironment,
        agent_factory: Callable[[str], DonorGameAgent],
        survivor_ratio: float = 0.5,
        num_agents: int = 4,
        num_generations: int = 10,
        rounds_per_generation: int = 5,
        stats_dir: str = "experiment_stats",
        model_name: str = "unknown",
        profile_name: str = "unknown",
        run_index: int = 0
    ):
        self.env = env
        self.agent_factory = agent_factory
        self.survivor_ratio = survivor_ratio
        self.num_agents = num_agents
        self.num_generations = num_generations
        self.rounds_per_generation = rounds_per_generation
        self.model_name = model_name.replace("/", "_")  # Clean model name for filesystem
        self.profile_name = profile_name
        self.run_index = run_index
        
        # Set up stats logging
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        experiment_dir = f"{model_name}_{profile_name}_run{run_index:02d}_{timestamp}"
        self.stats_dir = Path(stats_dir) / experiment_dir
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats_file = self.stats_dir / "experiment_stats.csv"
        self.agent_history_file = self.stats_dir / "agent_history.csv"
        
        # Create experiment logger
        self.logger = logging.getLogger("experiment")
        
        # Track agent histories
        self.agent_histories: Dict[str, AgentHistory] = {}
        
        # Set up donation tracking
        self.env.set_donation_callback(self._update_agent_stats)
    
    def _track_agent_history(self, agent: DonorGameAgent, generation: int) -> None:
        """Track an agent's history."""
        agent_id = agent.id
        
        # Create new history entry for new agents
        if agent_id not in self.agent_histories:
            self.agent_histories[agent_id] = AgentHistory(
                agent_id=agent_id,
                original_generation=generation,
                generations_survived=1,
                total_donations_made=0,
                total_amount_donated=0.0,
                total_amount_received=0.0,
                total_punishments_given=0,
                total_punishments_received=0,
                total_amount_spent_on_punishment=0.0,
                total_amount_lost_to_punishment=0.0,
                strategies=[agent.strategy],
                resources_over_time=[self.env.state.agent_resources[agent_id]]
            )
        else:
            # Update existing history
            history = self.agent_histories[agent_id]
            history.generations_survived += 1
            history.strategies.append(agent.strategy)
            history.resources_over_time.append(self.env.state.agent_resources[agent_id])
    
    def _update_agent_stats(self, donor_id: str, recipient_id: str, amount: float, punishment_amount: float) -> None:
        """Update agent statistics after a donation and/or punishment."""
        if donor_id in self.agent_histories:
            history = self.agent_histories[donor_id]
            if amount > 0:
                history.total_donations_made += 1
                history.total_amount_donated += amount
            if punishment_amount > 0:
                history.total_punishments_given += 1
                history.total_amount_spent_on_punishment += punishment_amount
        
        if recipient_id in self.agent_histories:
            history = self.agent_histories[recipient_id]
            if amount > 0:
                history.total_amount_received += amount * self.env.donation_mult
            if punishment_amount > 0:
                history.total_punishments_received += 1
                history.total_amount_lost_to_punishment += punishment_amount * 2  # Punishment multiplier
    
    def _log_agent_histories(self) -> None:
        """Log agent histories to a separate CSV file."""
        with open(self.agent_history_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                "AgentID", "OriginalGeneration", "GenerationsSurvived",
                "TotalDonationsMade", "TotalAmountDonated", "TotalAmountReceived",
                "TotalPunishmentsGiven", "TotalPunishmentsReceived",
                "TotalAmountSpentOnPunishment", "TotalAmountLostToPunishment",
                "AverageDonationAmount", "AveragePunishmentAmount",
                "ResourcesOverTime", "Strategies"
            ])
            
            # Write agent histories
            for history in self.agent_histories.values():
                avg_donation = (history.total_amount_donated / history.total_donations_made 
                              if history.total_donations_made > 0 else 0)
                avg_punishment = (history.total_amount_spent_on_punishment / history.total_punishments_given
                                if history.total_punishments_given > 0 else 0)
                
                writer.writerow([
                    history.agent_id,
                    history.original_generation,
                    history.generations_survived,
                    history.total_donations_made,
                    f"{history.total_amount_donated:.2f}",
                    f"{history.total_amount_received:.2f}",
                    history.total_punishments_given,
                    history.total_punishments_received,
                    f"{history.total_amount_spent_on_punishment:.2f}",
                    f"{history.total_amount_lost_to_punishment:.2f}",
                    f"{avg_donation:.2f}",
                    f"{avg_punishment:.2f}",
                    ",".join(f"{r:.2f}" for r in history.resources_over_time),
                    "|".join(history.strategies)
                ])
        
        # Log summary to console
        self.logger.info("\n=== Agent History Summary ===")
        for history in sorted(self.agent_histories.values(), 
                            key=lambda h: h.generations_survived, reverse=True):
            self.logger.info(f"\nAgent {history.agent_id}:")
            self.logger.info(f"  Original Generation: {history.original_generation}")
            self.logger.info(f"  Generations Survived: {history.generations_survived}")
            self.logger.info("\n  Donation Stats:")
            self.logger.info(f"    Total Donations Made: {history.total_donations_made}")
            self.logger.info(f"    Total Amount Donated: {history.total_amount_donated:.2f}")
            self.logger.info(f"    Total Amount Received: {history.total_amount_received:.2f}")
            if history.total_donations_made > 0:
                avg_donation = history.total_amount_donated / history.total_donations_made
                self.logger.info(f"    Average Donation: {avg_donation:.2f}")
            
            self.logger.info("\n  Punishment Stats:")
            self.logger.info(f"    Total Punishments Given: {history.total_punishments_given}")
            self.logger.info(f"    Total Punishments Received: {history.total_punishments_received}")
            self.logger.info(f"    Amount Spent on Punishment: {history.total_amount_spent_on_punishment:.2f}")
            self.logger.info(f"    Amount Lost to Punishment: {history.total_amount_lost_to_punishment:.2f}")
            if history.total_punishments_given > 0:
                avg_punishment = history.total_amount_spent_on_punishment / history.total_punishments_given
                self.logger.info(f"    Average Punishment: {avg_punishment:.2f}")
            
            self.logger.info(f"\n  Final Resources: {history.resources_over_time[-1]:.2f}")
        self.logger.info("=" * 40)
    
    def run(self) -> None:
        """Executes the experiment for the specified number of generations."""
        asyncio.run(self.run_async())
    
    async def run_async(self) -> None:
        """Executes the experiment asynchronously."""
        self.logger.info("Starting experiment")
        
        # Initialize first generation
        await self._initialize_generation(generation=1)
        
        # Log experiment configuration
        self._log_experiment_config()
        
        # Initialize stats file with header
        self._initialize_stats_file()
        
        # Run for specified number of generations
        for gen in range(1, self.num_generations + 1):
            self.logger.info(f"Starting generation {gen}")
            
            # Track agent histories at start of generation
            for agent in self.env.agents:
                self._track_agent_history(agent, gen)
            
            # Run all rounds in this generation
            await self._run_generation(gen)
            
            # Calculate and log statistics
            stats = self._calculate_generation_stats(gen)
            self._log_generation_stats(gen, stats)
            
            if gen < self.num_generations:
                # Select survivors and get their strategies
                survivors = self._select_survivors()
                survivor_advice = self._get_survivor_advice(survivors)
                
                # Initialize next generation
                await self._initialize_generation(gen + 1, survivor_advice)
        
        # Log final agent histories
        self._log_agent_histories()
    
    async def _initialize_generation(self, generation: int, survivor_advice: str = "") -> None:
        """Initialize a new generation of agents."""
        self.logger.info(f"Initializing generation {generation}")
        
        # Get surviving agents and their resources from previous generation if not first generation
        surviving_agents = []
        surviving_resources = {}
        survivor_ids = []
        if generation > 1:
            survivor_ids = self._select_survivors()
            surviving_agents = [agent for agent in self.env.agents if agent.id in survivor_ids]
            surviving_resources = {
                agent.id: self.env.state.agent_resources[agent.id]
                for agent in surviving_agents
            }
            self.logger.info(f"Reusing {len(surviving_agents)} surviving agents: {[a.id for a in surviving_agents]}")
        
        # Calculate how many new agents we need
        num_new_agents = self.num_agents - len(surviving_agents)
        
        # Create new agents
        new_agents = [
            self.agent_factory(f"{generation}_{i}")
            for i in range(num_new_agents)
        ]
        
        self.logger.info(f"Created {len(new_agents)} new agents: {[a.id for a in new_agents]}")
        
        # Reset environment but keep resources for survivors
        self.env.reset(keep_resources=True, survivor_ids=survivor_ids)
        
        # Generate strategies concurrently for all agents
        self.logger.info("Generating strategies in parallel...")
        start_time = time.time()
        
        await asyncio.gather(
            *[agent.generate_strategy(generation, survivor_advice) 
              for agent in surviving_agents + new_agents]
        )
        
        end_time = time.time()
        self.logger.info(f"All strategies generated in {end_time - start_time:.2f} seconds")
        
        # Add all agents to environment, preserving resources for survivors
        for agent in surviving_agents:
            self.env.add_agent(agent, initial_resources=surviving_resources[agent.id])
        for agent in new_agents:
            self.env.add_agent(agent)  # New agents get default initial balance
    
    async def _run_generation(self, generation: int) -> None:
        """Run all rounds in current generation."""
        for round_num in range(self.rounds_per_generation):
            self.logger.info(f"Generation {generation}, Round {round_num + 1}/{self.rounds_per_generation}")
            await self.env.step()
    
    def _select_survivors(self) -> List[str]:
        """Select top performing agents to survive to next generation."""
        num_survivors = int(self.num_agents * self.survivor_ratio)
        return self.env.get_top_agents(num_survivors)
    
    def _get_survivor_advice(self, survivors: List[str]) -> str:
        """Get advice from surviving agents for the next generation."""
        advice = []
        for agent in self.env.agents:
            if agent.id in survivors:
                resources = self.env.state.agent_resources[agent.id]
                advice.append(f"Agent {agent.id} ({resources:.2f} resources): {agent.strategy}")
        
        return "Successful strategies from previous generation:\n" + "\n".join(advice)
    
    def _calculate_generation_stats(self, generation: int) -> GenerationStats:
        """Calculate statistics for the current generation."""
        resources = list(self.env.state.agent_resources.values())
        total_resources = sum(resources)
        avg_resources = total_resources / len(resources)
        
        # Calculate standard deviation
        squared_diff_sum = sum((r - avg_resources) ** 2 for r in resources)
        std_dev = math.sqrt(squared_diff_sum / len(resources))
        
        # Calculate resource inequality
        resource_inequality = max(resources) - min(resources)
        
        # Calculate success rate
        total_donations = (self.env.state.successful_donations + 
                         self.env.state.failed_donations)
        success_rate = (
            (self.env.state.successful_donations / total_donations * 100)
            if total_donations > 0 else 0.0
        )
        
        return GenerationStats(
            total_resources=total_resources,
            avg_resources=avg_resources,
            std_dev=std_dev,
            resource_inequality=resource_inequality,
            successful_donations=self.env.state.successful_donations,
            failed_donations=self.env.state.failed_donations,
            success_rate=success_rate
        )
    
    def _log_generation_stats(self, generation: int, stats: GenerationStats) -> None:
        """Log statistics for the current generation."""
        # Get top performer information
        top_agents = self.env.get_top_agents(1)
        top_agent = next((a for a in self.env.agents if a.id == top_agents[0]), None)
        top_resources = self.env.state.agent_resources[top_agent.id] if top_agent else 0
        
        # Print to console
        self.logger.info(f"\n=== Generation {generation} Statistics ===")
        self.logger.info("Resource Metrics:")
        self.logger.info(f"  Total Resources: {stats.total_resources:.2f}")
        self.logger.info(f"  Average Resources: {stats.avg_resources:.2f}")
        self.logger.info(f"  Standard Deviation: {stats.std_dev:.2f}")
        self.logger.info(f"  Resource Inequality (max-min): {stats.resource_inequality:.2f}")
        self.logger.info("\nDonation Metrics:")
        self.logger.info(f"  Successful Donations: {stats.successful_donations}")
        self.logger.info(f"  Failed Donations: {stats.failed_donations}")
        self.logger.info(f"  Success Rate: {stats.success_rate:.1f}%")
        if top_agent:
            self.logger.info("\nTop Performer:")
            self.logger.info(f"  Agent: {top_agent.id}")
            self.logger.info(f"  Resources: {top_resources:.2f}")
            self.logger.info(f"  Model: {top_agent.model}")
            self.logger.info(f"  Cultural Profile: {top_agent.cultural_profile}")
        self.logger.info("=" * 40)
        
        # Write to CSV
        with open(self.stats_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                generation,
                f"{stats.total_resources:.2f}",
                f"{stats.avg_resources:.2f}",
                f"{stats.std_dev:.2f}",
                f"{stats.resource_inequality:.2f}",
                stats.successful_donations,
                stats.failed_donations,
                f"{stats.success_rate:.1f}",
                top_agent.id if top_agent else "",
                f"{top_resources:.2f}",
                top_agent.model if top_agent else "",
                str(top_agent.cultural_profile) if top_agent else ""
            ])
    
    def _initialize_stats_file(self) -> None:
        """Initialize the stats file with header and experiment configuration."""
        with open(self.stats_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write experiment configuration
            writer.writerow(["Experiment Configuration"])
            writer.writerow(["Number of Agents", self.num_agents])
            writer.writerow(["Number of Generations", self.num_generations])
            writer.writerow(["Rounds per Generation", self.rounds_per_generation])
            writer.writerow(["Survivor Ratio", self.survivor_ratio])
            writer.writerow(["Initial Balance", self.env.initial_balance])
            writer.writerow(["Donation Multiplier", self.env.donation_mult])
            writer.writerow([])
            
            # Write agent information
            writer.writerow(["Agent Configurations"])
            for agent in self.env.agents:
                writer.writerow([
                    f"Agent {agent.id}",
                    f"Model: {agent.model}",
                    f"Cultural Profile: {agent.cultural_profile.to_csv_str()}"
                ])
            writer.writerow([])
            
            # Write stats header
            writer.writerow([
                "Generation",
                "TotalResources",
                "AverageResources",
                "StandardDeviation",
                "ResourceInequality",
                "SuccessfulDonations",
                "FailedDonations",
                "SuccessRate",
                "TopPerformer",
                "TopPerformerResources",
                "TopPerformerModel",
                "TopPerformerProfile"
            ])
    
    def _log_experiment_config(self) -> None:
        """Log the experiment configuration."""
        self.logger.info("\n=== Experiment Configuration ===")
        self.logger.info(f"Number of Agents: {self.num_agents}")
        self.logger.info(f"Number of Generations: {self.num_generations}")
        self.logger.info(f"Rounds per Generation: {self.rounds_per_generation}")
        self.logger.info(f"Survivor Ratio: {self.survivor_ratio}")
        self.logger.info(f"Initial Balance: {self.env.initial_balance}")
        self.logger.info(f"Donation Multiplier: {self.env.donation_mult}")
        
        self.logger.info("\nAgent Configurations:")
        for agent in self.env.agents:
            self.logger.info(f"Agent {agent.id}:")
            self.logger.info(f"  Model: {agent.model}")
            self.logger.info(f"  Cultural Profile: {agent.cultural_profile}")
        self.logger.info("=" * 40)

# Example usage
if __name__ == "__main__":
    from profiles import create_example_profiles
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create example cultural profiles
    nordic, east_asian = create_example_profiles()
    
    # Create environment
    env = DonorGameEnvironment(rounds_per_gen=5)
    
    # Create agent factory
    def agent_factory(agent_id: str) -> DonorGameAgent:
        # Alternate between Nordic and East Asian profiles
        profile = nordic if int(agent_id.split('_')[1]) % 2 == 0 else east_asian
        return DonorGameAgent(profile, agent_id)
    
    # Create and run experiment
    experiment = DonorGameExperiment(
        env=env,
        agent_factory=agent_factory,
        survivor_ratio=0.5,
        num_agents=4,
        num_generations=10,
        rounds_per_generation=5,
        model_name="example_model",
        profile_name="example_profile",
        run_index=0
    )
    
    experiment.run() 