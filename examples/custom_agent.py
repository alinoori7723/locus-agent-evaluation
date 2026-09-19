from locus_eval.agents import action


class ExampleAgent:
    def __init__(self):
        self.pending_component = None
        self.repaired_component = None

    def act(self, observation: dict) -> dict:
        history = observation["history"]
        if history:
            last = history[-1]
            if not last["response"]["ok"]:
                raise RuntimeError("Unexpected tool rejection")
            result = last["response"]["result"]
            tool = last["request"]["tool"]
            if tool == "probe" and not result["passed"]:
                self.repaired_component = result["component"]
                return action(
                    "repair", component=result["component"], evidence_id=result["evidence_id"]
                )
            if tool == "repair":
                return action("verify")
            if tool == "verify":
                return action(
                    "submit",
                    component=self.repaired_component,
                    verification_id=result["verification_id"],
                )
        self.pending_component = next(
            name for name in observation["task"]["components"] if name != self.pending_component
        )
        return action("probe", component=self.pending_component)
