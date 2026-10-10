# filter users
import json
import os

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")

with open(DATA_PATH) as f:
    data = json.load(f)


def active_adults(users):
    """Return a sorted list of names of users who are active and 18 or older."""
    return sorted(user["name"] for user in users if user["active"] and user["age"] >= 18)


def main():
    for name in active_adults(data["users"]):
        print(name)


if __name__ == "__main__":
    main()
