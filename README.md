# Cultural Agents Donor Game

This project implements a multi-agent simulation of the Donor Game using Large Language Models (LLMs) as agents with distinct cultural profiles based on Hofstede's Cultural Dimensions Theory.

## Overview

The simulation explores how cultural differences influence cooperation and resource-sharing behavior in a modified version of the Donor Game. Each agent is imbued with cultural characteristics based on Hofstede's six cultural dimensions:

1. Power Distance (Hierarchical vs. Egalitarian)
2. Individualism vs. Collectivism
3. Masculinity vs. Femininity
4. Uncertainty Avoidance (Structured vs. Flexible)
5. Time Orientation (Long-term vs. Short-term)
6. Indulgence vs. Restraint

## Cultural Profiles

The simulation includes several pre-defined cultural profiles based on real-world cultural patterns:

- **Nordic** (Sweden, Denmark, Norway): Highly egalitarian, individualistic, and feminine-oriented society
- **East Asian** (Japan, China, South Korea): Hierarchical, collectivist, with high uncertainty avoidance
- **Western** (US, UK, Australia): Individualistic, egalitarian, with high indulgence
- **Arab**: Hierarchical, collectivist, with strong uncertainty avoidance
- **Latin American**: Collectivist, hierarchical, with varying masculinity levels

## The Donor Game

In each round of the game, agents:

1. Receive an initial balance of resources
2. Can choose to donate resources to other agents
3. Donated resources are multiplied by a factor (donation multiplier)
4. Make decisions based on their cultural profile and past interactions

## Usage

```bash
# Run with default settings (mixed cultural profiles)
python run_experiment.py

# Run with specific cultural profile
python run_experiment.py --profile-type nordic
python run_experiment.py --profile-type east_asian
python run_experiment.py --profile-type western
python run_experiment.py --profile-type arab
python run_experiment.py --profile-type latin_american

# Run with custom parameters
python run_experiment.py \
    --profile-type mixed \
    --num-agents 10 \
    --num-generations 5 \
    --rounds-per-generation 20 \
    --initial-balance 100 \
    --donation-multiplier 2.0
```

### Command Line Arguments

- `--profile-type, -p`: Cultural profile for agents (default: "mixed")
- `--model, -m`: LLM model to use (default: "claude")
- `--num-agents, -n`: Number of agents per generation
- `--num-generations, -g`: Number of generations to run
- `--rounds-per-generation, -r`: Number of rounds per generation
- `--survivor-ratio, -s`: Fraction of agents that survive to next generation
- `--initial-balance, -b`: Initial resource balance for each agent
- `--donation-multiplier, -d`: Multiplier for donations
- `--stats-dir`: Directory for storing experiment statistics
- `--run-index`: Index of this run (for unique directory naming)
- `--verbose, -v`: Enable verbose logging

## Project Structure

- `profiles.py`: Defines cultural profiles using Hofstede's dimensions
- `donor_game.py`: Implements the Donor Game environment and agent logic
- `experiment.py`: Manages experiment execution and data collection
- `run_experiment.py`: CLI interface for running experiments
- `constants.py`: Configuration constants and default values

## Requirements

- Python 3.8+
- Access to LLM APIs (Uses litellm internally so any OpenAI compatible API will work)
- Required Python packages (currently only litellm is needed, see requirements.txt)

## Data Collection

Experiment results are saved in the specified stats directory, including:

- Agent interactions and decisions
- Resource distribution over time
- Cultural profile performance metrics
- Generation-by-generation evolution data

## License

MIT
