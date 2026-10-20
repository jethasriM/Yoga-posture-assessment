"""Smoke test for the alignment feedback module."""

import json

from src.alignment_feedback import get_alignment_feedback


def main():
    pose_name = "Adho Mukha Svanasana"

    angles = {
        "left_knee": 160.75,
        "right_knee": 154.78,
        "left_elbow": 140.0,
        "right_elbow": 175.32,
    }

    feedback = get_alignment_feedback(pose_name, angles)

    print("\nPose:", pose_name)
    print("\nJoint angles:")
    print(json.dumps(angles, indent=2))

    print("\nGenerated feedback:")
    print(json.dumps(feedback, indent=2))


if __name__ == "__main__":
    main()