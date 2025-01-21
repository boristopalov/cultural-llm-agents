"""Constants shared across the donor game experiment."""

# Game mechanics
INITIAL_BALANCE = 10.0
DONATION_MULTIPLIER = 2.0
PUNISHMENT_MULTIPLIER = 2.0  # Punishment costs x to remove 2x from target
ROUNDS_PER_GENERATION = 10
MEMORY_SIZE = 100

# LLM prompts
SYSTEM_PROMPT = """Each player is given an initial endowment of 10 units of a resource. In each round, you are randomly paired with another individual. One of you is a donor, and the other is a recipient. The donor has two possible actions:
1. DONATE: Give up some amount of resource. The recipient receives 2x the number of units that the donor gave up.
2. PUNISH: Spend x units to remove 2x units from the recipient's resources.

You can both donate and punish in the same round, but the total cost cannot exceed your current resources. If you were a donor in one round, you will be a recipient in the next round, and vice versa. Your goal is to maximize the number of units you have after the final round. After the game has finished, the best-performing half of agents will survive to the next generation, and continue playing."""

STRATEGY_PROMPT_TEMPLATE = """Your name is {name}.
{context}
As a donor, you will receive limited information about the recipient and their recent interactions: You will see what the recipient (let's call them A) did in the immediately preceding round only. You will also see what A's partner from that round (let's call them B) did in their preceding round. This chain continues for at most three rounds back. Important: You do NOT have access to multiple rounds of information about A directly. You only know what A did in the single most recent round.

Example (assuming it's now round 4): You are paired with A. You know: In round 3, A donated X% and spent Y% on punishment to B. You also know: In round 2, B donated Z% and spent W% on punishment to C. Remember: This is all the information you have. You don't know what A did in rounds 1 or 2, only in round 3.

In the first round, there will be no information about the recipient's previous behavior - think carefully about your actions. Consider both donation and punishment options. Before formulating your strategy, briefly think step by step about what would be a successful strategy in this game. Then describe your strategy briefly without explanation in one sentence that starts: My strategy will be."""

DONATION_PROMPT_TEMPLATE = """Your name is {name}. As you will recall, here is the strategy you decided to follow: "{strategy}"

This is generation {generation}. It is now round {round}. In this round, you have been paired with {recipient}. They currently have {recipient_resources:.2f} units of the valuable resource.

{recipient_history}

You currently have {donor_resources:.2f} units of the valuable resource.

You must decide:
1. How many units to donate (recipient will receive 2x this amount)
2. How many units to spend on punishment (recipient will lose 2x this amount)

The total cost of donation + punishment cannot exceed your current resources.

Very briefly think step by step about how you apply your strategy in this situation and then provide your answer in the following format:

DONATION: <amount>
PUNISHMENT: <amount>"""

# Default experiment parameters
DEFAULT_NUM_AGENTS = 4
DEFAULT_NUM_GENERATIONS = 10
DEFAULT_SURVIVOR_RATIO = 0.5
DEFAULT_STATS_DIR = "experiment_stats" 