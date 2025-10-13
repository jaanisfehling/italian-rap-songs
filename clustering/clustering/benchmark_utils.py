from typing import Any


def print_best(best: dict[Any, tuple[str, float]]):
    print("Winners:")
    longest = max(map(len, best))
    for d, r in best.items():
        print(f"{d}{" " * (longest - len(d))}  {r[0]}")

    print("\nMost frequent winners:")
    ms = [r[0] for r in best.values()]
    counts = {m: ms.count(m) for m in ms}
    s_counts = [
        (m, c)
        for m, c in sorted(counts.items(), key=lambda item: item[1], reverse=True)
    ]
    if len(s_counts) >= 3:
        print(
            f"{s_counts[0][0]} ({s_counts[0][1]}x), {s_counts[1][0]} ({s_counts[1][1]}x), {s_counts[2][0]} ({s_counts[2][1]}x)"
        )
    elif len(s_counts) == 2:
        print(
            f"{s_counts[0][0]} ({s_counts[0][1]}x), {s_counts[1][0]} ({s_counts[1][1]}x)"
        )
    else:
        print(f"{s_counts[0][0]} ({s_counts[0][1]}x)")
