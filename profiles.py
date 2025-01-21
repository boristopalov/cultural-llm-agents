from enum import Enum, auto
from typing import Dict, List
from dataclasses import dataclass

class PowerDistance(Enum):
    EGALITARIAN = auto()      
    MODERATE = auto()         
    HIERARCHICAL = auto()     

class Individualism(Enum):
    COLLECTIVIST = auto()     
    BALANCED = auto()         
    INDIVIDUALIST = auto()    

class Masculinity(Enum):
    FEMININE = auto()         
    MODERATE = auto()         
    MASCULINE = auto()        

class UncertaintyAvoidance(Enum):
    FLEXIBLE = auto()         
    MODERATE = auto()         
    STRUCTURED = auto()       

class TimeOrientation(Enum):
    SHORT_TERM = auto()       
    BALANCED = auto()         
    LONG_TERM = auto()        

class Indulgence(Enum):
    RESTRAINT = auto()        
    BALANCED = auto()         
    INDULGENT = auto()       

@dataclass
class CulturalProfile:
    """
    Defines a cultural agent profile using Hofstede's 6 dimensions.
    Generates system prompts that instruct LLMs to embody specific cultural values.
    """
    
    power_distance: PowerDistance
    individualism: Individualism
    masculinity: Masculinity
    uncertainty_avoidance: UncertaintyAvoidance
    time_orientation: TimeOrientation
    indulgence: Indulgence

    def __str__(self) -> str:
        """Returns a human-readable string representation of the cultural profile."""
        return (
            f"Cultural Profile:\n"
            f"  Power Distance: {self.power_distance.name}\n"
            f"  Individualism: {self.individualism.name}\n"
            f"  Masculinity: {self.masculinity.name}\n"
            f"  Uncertainty Avoidance: {self.uncertainty_avoidance.name}\n"
            f"  Time Orientation: {self.time_orientation.name}\n"
            f"  Indulgence: {self.indulgence.name}"
        )
    
    def __repr__(self) -> str:
        """Returns a string that could be used to recreate the object."""
        return (
            f"CulturalProfile("
            f"power_distance={self.power_distance.name}, "
            f"individualism={self.individualism.name}, "
            f"masculinity={self.masculinity.name}, "
            f"uncertainty_avoidance={self.uncertainty_avoidance.name}, "
            f"time_orientation={self.time_orientation.name}, "
            f"indulgence={self.indulgence.name})"
        )

    def to_csv_str(self) -> str:
        """Returns a single-line string representation suitable for CSV."""
        return (
            f"[PowerDistance:{self.power_distance.name}, "
            f"Individualism:{self.individualism.name}, "
            f"Masculinity:{self.masculinity.name}, "
            f"UncertaintyAvoidance:{self.uncertainty_avoidance.name}, "
            f"TimeOrientation:{self.time_orientation.name}, "
            f"Indulgence:{self.indulgence.name}]"
        )

    def generate_system_prompt(self) -> str:
        """
        Generates a comprehensive system prompt that defines the agent's cultural values
        and provides concrete examples of their application.
        """
        values_description = self._get_values_description()
        behavior_examples = self._get_behavior_examples()
        
        prompt = f"""You are an agent with the following cultural values and orientations:

{values_description}

Here's how these values manifest in various situations:
{behavior_examples}

When responding to queries and making decisions, you should embody these cultural values consistently. Your responses should reflect these orientations while remaining helpful and engaging."""

        return prompt

    def _get_values_description(self) -> str:
        """Generates detailed descriptions of each cultural value."""
        descriptions = []
        
        # Power Distance
        pd_desc = "1. Hierarchy and Authority: "
        if self.power_distance == PowerDistance.HIERARCHICAL:
            pd_desc += "You strongly respect hierarchical order and formal authority. Status differences should be visible and respected."
        elif self.power_distance == PowerDistance.EGALITARIAN:
            pd_desc += "You believe in flat hierarchies and equal treatment regardless of formal position. Power differences should be minimized."
        else:
            pd_desc += "You maintain a balanced view of hierarchy, adapting to the context."
        descriptions.append(pd_desc)

        # Individualism
        ind_desc = "2. Individual vs. Group: "
        if self.individualism == Individualism.INDIVIDUALIST:
            ind_desc += "You prioritize individual goals and personal achievement. Direct communication and personal opinions are valued."
        elif self.individualism == Individualism.COLLECTIVIST:
            ind_desc += "You prioritize group harmony and collective goals. Relationships and group consensus are essential."
        else:
            ind_desc += "You balance individual needs with group obligations."
        descriptions.append(ind_desc)

        # Masculinity
        masc_desc = "3. Competition vs. Consensus: "
        if self.masculinity == Masculinity.MASCULINE:
            masc_desc += "You value achievement, assertiveness, and competitive success. Excellence should be rewarded and recognized."
        elif self.masculinity == Masculinity.FEMININE:
            masc_desc += "You prioritize cooperation, modesty, and quality of life. Consensus and harmony are key goals."
        else:
            masc_desc += "You balance competition with cooperation based on context."
        descriptions.append(masc_desc)

        # Uncertainty Avoidance
        ua_desc = "4. Ambiguity Tolerance: "
        if self.uncertainty_avoidance == UncertaintyAvoidance.STRUCTURED:
            ua_desc += "You prefer clear structure and detailed plans. Uncertainty makes you uncomfortable and should be minimized."
        elif self.uncertainty_avoidance == UncertaintyAvoidance.FLEXIBLE:
            ua_desc += "You are comfortable with ambiguity and flexible plans. Uncertainty is natural and can bring opportunities."
        else:
            ua_desc += "You handle uncertainty with measured caution, seeking clarity when important."
        descriptions.append(ua_desc)

        # Time Orientation
        time_desc = "5. Time Perspective: "
        if self.time_orientation == TimeOrientation.LONG_TERM:
            time_desc += "You focus on long-term success and future outcomes. Traditional practices should adapt to changing circumstances."
        elif self.time_orientation == TimeOrientation.SHORT_TERM:
            time_desc += "You emphasize immediate results and respect for tradition. Stability and current obligations are important."
        else:
            time_desc += "You balance short-term needs with long-term considerations."
        descriptions.append(time_desc)

        # Indulgence
        ind_desc = "6. Gratification Orientation: "
        if self.indulgence == Indulgence.INDULGENT:
            ind_desc += "You believe in enjoying life and following natural human desires. Fun and personal freedoms are important."
        elif self.indulgence == Indulgence.RESTRAINT:
            ind_desc += "You value moderation and believe desires should be regulated by strict social norms."
        else:
            ind_desc += "You balance enjoyment with appropriate restraint based on context."
        descriptions.append(ind_desc)

        return "\n\n".join(descriptions)

    def _get_behavior_examples(self) -> str:
        """Provides concrete examples of how the cultural values affect behavior."""
        examples = []

        # Decision-making example
        decision_example = self._generate_decision_making_example()
        examples.append(f"Decision-making: {decision_example}")

        # Communication example
        communication_example = self._generate_communication_example()
        examples.append(f"Communication: {communication_example}")

        # Conflict resolution example
        conflict_example = self._generate_conflict_example()
        examples.append(f"Conflict Resolution: {conflict_example}")

        return "\n\n".join(examples)

    def _generate_decision_making_example(self) -> str:
        """Generates an example of decision-making behavior based on cultural values."""
        if (self.power_distance == PowerDistance.HIERARCHICAL and 
            self.individualism == Individualism.COLLECTIVIST):
            return "When making decisions, you prefer to seek guidance from authority figures and ensure group consensus before proceeding."
        elif (self.power_distance == PowerDistance.EGALITARIAN and 
              self.individualism == Individualism.INDIVIDUALIST):
            return "You make decisions independently, valuing personal judgment and direct discussion with all stakeholders regardless of their position."
        return "You adapt your decision-making approach based on the situation, considering both authority and individual input."

    def _generate_communication_example(self) -> str:
        """Generates an example of communication style based on cultural values."""
        if (self.individualism == Individualism.INDIVIDUALIST and 
            self.uncertainty_avoidance == UncertaintyAvoidance.FLEXIBLE):
            return "You communicate directly and explicitly, comfortable with challenging ideas and expressing disagreement openly."
        elif (self.individualism == Individualism.COLLECTIVIST and 
              self.uncertainty_avoidance == UncertaintyAvoidance.STRUCTURED):
            return "You communicate indirectly and carefully, prioritizing harmony and avoiding potential conflicts or ambiguity."
        return "You adjust your communication style to the context, balancing directness with tact."

    def _generate_conflict_example(self) -> str:
        """Generates an example of conflict resolution based on cultural values."""
        if (self.masculinity == Masculinity.FEMININE and 
            self.individualism == Individualism.COLLECTIVIST):
            return "In conflicts, you prioritize maintaining harmony and finding solutions that preserve relationships and group cohesion."
        elif (self.masculinity == Masculinity.MASCULINE and 
              self.individualism == Individualism.INDIVIDUALIST):
            return "You address conflicts directly, focusing on achieving clear resolution and establishing who is right based on facts."
        return "You handle conflicts by balancing the need for resolution with maintaining positive relationships."

# Example usage
def create_example_profiles():
    # Nordic profile (Sweden, Denmark, Norway)
    nordic = CulturalProfile(
        power_distance=PowerDistance.EGALITARIAN,
        individualism=Individualism.INDIVIDUALIST,
        masculinity=Masculinity.FEMININE,
        uncertainty_avoidance=UncertaintyAvoidance.MODERATE,
        time_orientation=TimeOrientation.BALANCED,
        indulgence=Indulgence.INDULGENT
    )
    
    # East Asian profile (Japan, China, South Korea)
    east_asian = CulturalProfile(
        power_distance=PowerDistance.HIERARCHICAL,
        individualism=Individualism.COLLECTIVIST,
        masculinity=Masculinity.MODERATE,
        uncertainty_avoidance=UncertaintyAvoidance.STRUCTURED,
        time_orientation=TimeOrientation.LONG_TERM,
        indulgence=Indulgence.RESTRAINT
    )

    # Western profile (US, UK, Australia)
    western = CulturalProfile(
        power_distance=PowerDistance.EGALITARIAN,
        individualism=Individualism.INDIVIDUALIST,
        masculinity=Masculinity.MASCULINE,
        uncertainty_avoidance=UncertaintyAvoidance.FLEXIBLE,
        time_orientation=TimeOrientation.SHORT_TERM,
        indulgence=Indulgence.INDULGENT
    )

    # Arab profile
    arab = CulturalProfile(
        power_distance=PowerDistance.HIERARCHICAL,
        individualism=Individualism.COLLECTIVIST,
        masculinity=Masculinity.MASCULINE,
        uncertainty_avoidance=UncertaintyAvoidance.STRUCTURED,
        time_orientation=TimeOrientation.SHORT_TERM,
        indulgence=Indulgence.RESTRAINT
    )

    # Latin American profile
    latin_american = CulturalProfile(
        power_distance=PowerDistance.HIERARCHICAL,
        individualism=Individualism.COLLECTIVIST,
        masculinity=Masculinity.MODERATE,
        uncertainty_avoidance=UncertaintyAvoidance.STRUCTURED,
        time_orientation=TimeOrientation.SHORT_TERM,
        indulgence=Indulgence.INDULGENT
    )
    
    return {
        "nordic": nordic,
        "east_asian": east_asian,
        "western": western,
        "arab": arab,
        "latin_american": latin_american
    }