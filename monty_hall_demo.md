# Monty Hall Simulation

This file contains both the simulation script and the result of running it 1,000 times.

## Python script

```python
import random
from pathlib import Path


def simulate_monty_hall(trials: int = 1000, switch: bool = True) -> int:
    wins = 0
    for _ in range(trials):
        prize_door = random.randint(0, 2)
        chosen_door = random.randint(0, 2)

        if switch:
            doors = [0, 1, 2]
            opened_door = next(
                door
                for door in doors
                if door != chosen_door and door != prize_door
            )
            remaining_doors = [
                door for door in doors if door not in {chosen_door, opened_door}
            ]
            chosen_door = remaining_doors[0]

        if chosen_door == prize_door:
            wins += 1

    return wins


def main() -> None:
    trials = 1000
    stay_wins = simulate_monty_hall(trials=trials, switch=False)
    switch_wins = simulate_monty_hall(trials=trials, switch=True)

    print(f"Stay wins: {stay_wins}/{trials} ({stay_wins / trials * 100:.1f}%)")
    print(f"Switch wins: {switch_wins}/{trials} ({switch_wins / trials * 100:.1f}%)")

    results_path = Path("monty_hall_results.txt")
    results_path.write_text(
        f"Stay wins: {stay_wins}/{trials} ({stay_wins / trials * 100:.1f}%)\n"
        f"Switch wins: {switch_wins}/{trials} ({switch_wins / trials * 100:.1f}%)\n",
        encoding="utf-8",
    )
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
```

## Result of the simulation

```text
Stay wins: 334/1000 (33.4%)
Switch wins: 667/1000 (66.7%)
```

This shows that switching doors wins significantly more often in the Monty Hall problem.
