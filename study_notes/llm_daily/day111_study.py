"""Day 111: Prepare a local or small-cloud deployment configuration."""

import json


def health_response(path):
    # A fixed health route gives orchestration systems a lightweight readiness contract distinct from application routes.
    return {"status": 200, "body": {"ok": True}} if path == "/health" else {"status": 404, "body": {"error": "not found"}}


def main():
    # Local binding restricts access to the host, while the cloud plan lists controls required for public exposure.
    deployment = {
        "local": {"bind": "127.0.0.1", "port": 8000, "command": "python app.py", "privacy": "requests stay on the host"},
        "cloud": {"bind": "managed HTTPS endpoint", "requirements": ["authentication", "rate limits", "secret management", "monitoring"]},
    }
    # Sorted JSON makes the printed configuration stable for comparison and review.
    print("deployment checklist:", json.dumps(deployment, sort_keys=True))
    print("local health route:", health_response("/health"))
    print("This is a deployment plan and route contract; launch a real service only after its controls are configured.")

# Deployment configuration makes runtime assumptions explicit, which helps reproduce the same service outside development.
if __name__ == "__main__":
    main()
