import json
from pathlib import Path


DATA_FILE = Path(__file__).with_name("data.json")


def active_adults(users):
    return sorted(
        user["name"]
        for user in users
        if user.get("active") and user.get("age", 0) >= 18
    )


def main():
    with DATA_FILE.open() as f:
        data = json.load(f)

    for name in active_adults(data.get("users", [])):
        print(name)


if __name__ == "__main__":
    main()
