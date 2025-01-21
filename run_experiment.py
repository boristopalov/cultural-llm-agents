#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
from profiles import create_example_profiles
from donor_game import DonorGameAgent, DonorGameEnvironment
from experiment import DonorGameExperiment
from constants import (
    DEFAULT_NUM_AGENTS, DEFAULT_NUM_GENERATIONS,
    DEFAULT_SURVIVOR_RATIO, DEFAULT_STATS_DIR,
    INITIAL_BALANCE, DONATION_MULTIPLIER, ROUNDS_PER_GENERATION
)

# Available models
MODELS = {
    "claude": "anthropic/claude-3-sonnet-20240229",
    "deepseek": "deepseek/deepseek-chat"
}

def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def create_agent_factory(profile_type: str, model: str):
    """Creates an agent factory based on the specified profile type."""
    profiles = create_example_profiles()
    
    # Get the full model name from our mapping
    model_name = MODELS.get(model, model)  # Use provided model string if not in mapping
    
    if profile_type == "mixed":
        profile_types = list(profiles.keys())
        return lambda agent_id: DonorGameAgent(
            profiles[profile_types[int(agent_id.split('_')[1]) % len(profile_types)]],
            agent_id,
            model=model_name
        )
    elif profile_type in profiles:
        return lambda agent_id: DonorGameAgent(profiles[profile_type], agent_id, model=model_name)
    else:
        raise ValueError(f"Unknown profile type: {profile_type}")

def main():
    parser = argparse.ArgumentParser(description="Run donor game experiments with cultural agents")
    
    # Model selection
    parser.add_argument("--model", "-m",
                       choices=list(MODELS.keys()) + ["custom"],
                       default="claude",
                       help="LLM model to use for agents")
    parser.add_argument("--custom-model",
                       help="Custom model identifier (only used if --model=custom)")
    
    # Experiment parameters
    parser.add_argument("--profile-type", "-p", 
                       choices=["nordic", "east_asian", "western", "arab", "latin_american", "mixed"],
                       default="mixed",
                       help="Cultural profile type for agents")
    parser.add_argument("--num-agents", "-n", 
                       type=int, 
                       default=DEFAULT_NUM_AGENTS,
                       help="Number of agents per generation")
    parser.add_argument("--num-generations", "-g", 
                       type=int, 
                       default=DEFAULT_NUM_GENERATIONS,
                       help="Number of generations to run")
    parser.add_argument("--rounds-per-generation", "-r", 
                       type=int, 
                       default=ROUNDS_PER_GENERATION,
                       help="Number of rounds per generation")
    parser.add_argument("--survivor-ratio", "-s", 
                       type=float, 
                       default=DEFAULT_SURVIVOR_RATIO,
                       help="Fraction of agents that survive to next generation")
    
    # Environment parameters
    parser.add_argument("--initial-balance", "-b", 
                       type=float, 
                       default=INITIAL_BALANCE,
                       help="Initial resource balance for each agent")
    parser.add_argument("--donation-multiplier", "-d", 
                       type=float, 
                       default=DONATION_MULTIPLIER,
                       help="Multiplier for donations")
    
    # Output parameters
    parser.add_argument("--stats-dir", 
                       type=str, 
                       default=DEFAULT_STATS_DIR,
                       help="Directory for storing experiment statistics")
    parser.add_argument("--run-index",
                       type=int,
                       default=0,
                       help="Index of this run (for unique directory naming)")
    parser.add_argument("--verbose", "-v", 
                       action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Determine which model to use
    model = args.custom_model if args.model == "custom" else args.model
    
    try:
        # Create environment
        env = DonorGameEnvironment(
            rounds_per_gen=args.rounds_per_generation,
            donation_mult=args.donation_multiplier,
            initial_balance=args.initial_balance
        )
        
        # Create agent factory with specified model
        agent_factory = create_agent_factory(args.profile_type, model)
        
        # Create and run experiment
        experiment = DonorGameExperiment(
            env=env,
            agent_factory=agent_factory,
            survivor_ratio=args.survivor_ratio,
            num_agents=args.num_agents,
            num_generations=args.num_generations,
            rounds_per_generation=args.rounds_per_generation,
            stats_dir=args.stats_dir,
            model_name=args.model,
            profile_name=args.profile_type,
            run_index=args.run_index
        )
        
        # Run experiment
        experiment.run()
        
        logging.info(f"Experiment completed. Statistics saved in {args.stats_dir}/")
        
    except Exception as e:
        logging.error(f"Error running experiment: {e}")
        raise

if __name__ == "__main__":
    main() 