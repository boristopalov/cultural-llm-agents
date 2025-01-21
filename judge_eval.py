from typing import Dict, List, Tuple
from enum import Enum, auto
import re
from profiles import CulturalProfile, PowerDistance, Individualism, Masculinity, UncertaintyAvoidance, TimeOrientation, Indulgence

class AdherenceLevel(Enum):
    """Represents how well a response adheres to a cultural dimension."""
    CONTRADICTS = 0.0      # Response actively contradicts the cultural value
    NEUTRAL = 0.3          # Response neither adheres to nor contradicts the value
    SOMEWHAT = 0.7         # Response somewhat demonstrates the cultural value
    STRONGLY = 1.0         # Response strongly demonstrates the cultural value

class JudgeEvaluator:
    """
    Creates prompts for and parses responses from a judge LLM that evaluates 
    how well responses adhere to specified cultural profiles.
    """
    
    def set_cultural_profile(self, cultural_profile: CulturalProfile):
        self.cultural_profile = cultural_profile

    def create_judge_prompt(self, response: str) -> str:
        """
        Creates a system prompt for the judge LLM to evaluate a response's cultural adherence.
        """
        target_values = self._get_target_values_description()
        
        prompt = f"""You are an objective judge evaluating how well a response adheres to specific cultural values based on Hofstede's 6 cultural dimensions.

The response you are evaluating should reflect the following cultural profile:

{target_values}

For the following response, evaluate how well it adheres to each dimension using these levels:
- CONTRADICTS (0.0): Response actively contradicts the cultural value
- NEUTRAL (0.3): Response neither adheres to nor contradicts the value
- SOMEWHAT (0.7): Response somewhat demonstrates the cultural value
- STRONGLY (1.0): Response strongly demonstrates the cultural value

Response to evaluate:
"{response}"

Provide your evaluation in the following format:
power_distance: <LEVEL>
individualism: <LEVEL>
masculinity: <LEVEL>
uncertainty_avoidance: <LEVEL>
time_orientation: <LEVEL>
indulgence: <LEVEL>

Explanation:
[Provide a brief explanation of your evaluation for each dimension]"""

        return prompt

    def _get_target_values_description(self) -> str:
        """Generates a description of the target cultural values."""
        descriptions = []
        
        # Power Distance
        pd_desc = "Power Distance: "
        if self.cultural_profile.power_distance == PowerDistance.HIERARCHICAL:
            pd_desc += "Should show strong respect for hierarchy and authority"
        elif self.cultural_profile.power_distance == PowerDistance.EGALITARIAN:
            pd_desc += "Should demonstrate equality and minimal power differences"
        else:
            pd_desc += "Should show moderate respect for hierarchy"
        descriptions.append(pd_desc)
        
        # Individualism
        ind_desc = "Individualism: "
        if self.cultural_profile.individualism == Individualism.INDIVIDUALIST:
            ind_desc += "Should prioritize individual goals and direct communication"
        elif self.cultural_profile.individualism == Individualism.COLLECTIVIST:
            ind_desc += "Should emphasize group harmony and collective goals"
        else:
            ind_desc += "Should balance individual and group needs"
        descriptions.append(ind_desc)
        
        # Masculinity
        masc_desc = "Masculinity: "
        if self.cultural_profile.masculinity == Masculinity.MASCULINE:
            masc_desc += "Should value competition, achievement, and success"
        elif self.cultural_profile.masculinity == Masculinity.FEMININE:
            masc_desc += "Should emphasize cooperation, modesty, and caring"
        else:
            masc_desc += "Should balance competition and cooperation"
        descriptions.append(masc_desc)
        
        # Uncertainty Avoidance
        ua_desc = "Uncertainty Avoidance: "
        if self.cultural_profile.uncertainty_avoidance == UncertaintyAvoidance.STRUCTURED:
            ua_desc += "Should prefer clear structure and minimize ambiguity"
        elif self.cultural_profile.uncertainty_avoidance == UncertaintyAvoidance.FLEXIBLE:
            ua_desc += "Should be comfortable with ambiguity and flexible approaches"
        else:
            ua_desc += "Should show moderate tolerance for uncertainty"
        descriptions.append(ua_desc)
        
        # Time Orientation
        time_desc = "Time Orientation: "
        if self.cultural_profile.time_orientation == TimeOrientation.LONG_TERM:
            time_desc += "Should focus on long-term outcomes and future planning"
        elif self.cultural_profile.time_orientation == TimeOrientation.SHORT_TERM:
            time_desc += "Should emphasize immediate results and traditions"
        else:
            time_desc += "Should balance short-term and long-term perspectives"
        descriptions.append(time_desc)
        
        # Indulgence
        ind_desc = "Indulgence: "
        if self.cultural_profile.indulgence == Indulgence.INDULGENT:
            ind_desc += "Should value enjoyment and natural human desires"
        elif self.cultural_profile.indulgence == Indulgence.RESTRAINT:
            ind_desc += "Should emphasize restraint and strict social norms"
        else:
            ind_desc += "Should balance enjoyment with moderation"
        descriptions.append(ind_desc)
        
        return "\n".join(descriptions)

    def parse_judge_response(self, judge_response: str) -> Tuple[float, Dict[str, float], str]:
        """
        Parses the judge LLM's response to extract adherence levels and explanation.
        
        Expected format:
        power_distance: LEVEL
        individualism: LEVEL
        masculinity: LEVEL
        uncertainty_avoidance: LEVEL
        time_orientation: LEVEL
        indulgence: LEVEL
        
        Explanation:
        [explanation text]
        """
        
        # Initialize scores dictionary
        dimension_scores = {}
        
        # Define regex patterns
        score_pattern = r"(\w+(?:_\w+)?)\s*:\s*(CONTRADICTS|NEUTRAL|SOMEWHAT|STRONGLY)"
        explanation_pattern = r"Explanation:\s*([\s\S]+)$"  # [\s\S]+ matches any character including newlines
        
        # Extract scores
        matches = re.finditer(score_pattern, judge_response)
        for match in matches:
            dimension = match.group(1).lower()
            level = match.group(2)
            
            # Convert level string to score
            score = {
                "CONTRADICTS": AdherenceLevel.CONTRADICTS.value,
                "NEUTRAL": AdherenceLevel.NEUTRAL.value,
                "SOMEWHAT": AdherenceLevel.SOMEWHAT.value,
                "STRONGLY": AdherenceLevel.STRONGLY.value
            }.get(level, AdherenceLevel.NEUTRAL.value)
            
            dimension_scores[dimension] = score
        
        # Verify all dimensions are present
        expected_dimensions = {"power_distance", "individualism", "masculinity", 
                             "uncertainty_avoidance", "time_orientation", "indulgence"}
        missing_dimensions = expected_dimensions - set(dimension_scores.keys())
        
        # Fill in missing dimensions with NEUTRAL
        for dimension in missing_dimensions:
            dimension_scores[dimension] = AdherenceLevel.NEUTRAL.value
        
        # Extract explanation
        explanation_match = re.search(explanation_pattern, judge_response)
        explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided"
        
        # Calculate overall score
        overall_score = sum(dimension_scores.values()) / len(dimension_scores)
        
        return overall_score, dimension_scores, explanation

# Example usage
if __name__ == "__main__":
    from profiles import create_example_profiles
    
    nordic = create_example_profiles()[0]  # Get Nordic profile
    
    # Example response to evaluate
    response = "Let's discuss this openly as equals. I think we should make a decision that works for everyone involved."
    
    # Create evaluator
    evaluator = JudgeEvaluator()
    evaluator.set_cultural_profile(nordic)
    
    # Get the prompt for the judge LLM
    judge_prompt = evaluator.create_judge_prompt(response)
    print("JUDGE PROMPT:")
    print("=============")
    print(judge_prompt)
    
    # Example of parsing a judge's response (using placeholder values)
    print("\nPARSED RESULTS:")
    print("==============")
    overall_score, dimension_scores, explanation = evaluator.parse_judge_response("")
    for dimension, score in dimension_scores.items():
        print(f"{dimension}: {score:.2f}")
