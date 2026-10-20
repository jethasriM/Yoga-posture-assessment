"""Rule-based yoga alignment feedback using estimated joint angles."""

from typing import Dict, List, Optional


def get_alignment_feedback(
    pose_name: str,
    angles: Dict[str, Optional[float]]
) -> List[dict]:
    """
    Generate preliminary alignment feedback from joint angles.

    angles example:
    {
        "left_knee": 165.0,
        "right_knee": 160.0,
        "left_elbow": 175.0,
        "right_elbow": 172.0
    }

    Angle thresholds are illustrative starting rules, not
    clinically or professionally validated alignment standards.
    """

    feedback = []
    pose = pose_name.strip().lower().replace("_", " ")

    # Skip missing or invalid measurements.
    valid_angles = {}

    for joint, angle in angles.items():
        if angle is None:
            continue

        try:
            value = float(angle)
        except (TypeError, ValueError):
            continue

        if 0 <= value <= 180:
            valid_angles[joint] = value

    # Preliminary rules for poses that often use extended limbs.
    extended_pose_keywords = (
        "downward facing dog",
        "adho mukha svanasana",
        "plank",
        "phalakasana",
    )

    if any(keyword in pose for keyword in extended_pose_keywords):
        for joint in ("left_elbow", "right_elbow"):
            angle = valid_angles.get(joint)

            if angle is not None and angle < 150:
                feedback.append({
                    "joint": joint,
                    "issue": "possible_elbow_bend",
                    "message": (
                        f"{joint.replace('_', ' ').title()}: "
                        "Your elbow appears bent. Check your arm "
                        "position and avoid forcing the joint."
                    ),
                    "measured_angle": round(angle, 2),
                })

    # Report left/right differences as observations, not proof of error.
    for left_joint, right_joint in (
        ("left_knee", "right_knee"),
        ("left_elbow", "right_elbow"),
    ):
        left_angle = valid_angles.get(left_joint)
        right_angle = valid_angles.get(right_joint)

        if left_angle is None or right_angle is None:
            continue

        difference = abs(left_angle - right_angle)

        if difference > 15:
            feedback.append({
                "joint": left_joint.replace("left_", ""),
                "issue": "left_right_angle_difference",
                "message": (
                    f"Your left and right joint angles differ by "
                    f"{difference:.1f} degrees. Check your alignment "
                    "and the camera view."
                ),
                "measured_difference": round(difference, 2),
            })

    if not feedback:
        feedback.append({
            "joint": "general",
            "issue": "no_rule_triggered",
            "message": (
                "No configured alignment warning was triggered. "
                "This does not guarantee that the pose is correct."
            ),
        })

    return feedback