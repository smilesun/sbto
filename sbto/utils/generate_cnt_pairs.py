def generate_mujoco_contact_pairs(obj_geoms):
    """
    Generate a MuJoCo XML contact section for the given object geoms.
    """

    robot_geoms = [
        ("right_foot2_obj", "right_foot2_collision"),
        ("left_foot2_obj", "left_foot2_collision"),
        ("right_shin_obj", "right_shin_collision"),
        ("left_shin_obj", "left_shin_collision"),
        ("right_thigh_obj", "right_thigh_collision"),
        ("left_thigh_obj", "left_thigh_collision"),
        ("right_hip_obj", "right_hip_collision"),
        ("left_hip_obj", "left_hip_collision"),
        ("torso_obj", "torso_collision"),
        ("pelvis_obj", "pelvis_collision"),
        ("head_obj", "head_collision"),
        ("left_elbow_yaw_obj", "left_elbow_yaw_collision"),
        ("right_elbow_yaw_obj", "right_elbow_yaw_collision"),
        ("left_shoulder_yaw_obj", "left_shoulder_yaw_collision"),
        ("right_shoulder_yaw_obj", "right_shoulder_yaw_collision"),
        ("left_hand_obj", "left_hand_collision"),
        ("right_hand_obj", "right_hand_collision"),
        ("left_wrist_obj", "left_wrist_collision"),
        ("right_wrist_obj", "right_wrist_collision"),
    ]

    xml_lines = []
    xml_lines.append("<mujoco>\n  <contact>")

    for obj in obj_geoms:
        xml_lines.append(f"\n    <!-- COLLISIONS WITH: {obj} -->")
        for pair_name, robot_geom in robot_geoms:
            name = f"{pair_name}_{obj}"
            xml_lines.append(
                f'    <pair name="{name}" geom1="{robot_geom}" geom2="{obj}" condim="1"/>'
            )

    xml_lines.append("\n  </contact>\n</mujoco>")
    return "\n".join(xml_lines)


# Example usage
obj_list = ["obj", "back_left_leg", "back_right_leg", "front_left_leg", "front_right_leg", "backrest"]
obj_list = ["obj", "tube_0", "tube_1", "tube_2", "tube_3", "planck_0", "planck_1"]
xml_output = generate_mujoco_contact_pairs(obj_list)

# Save to XML file
output_filename = "contacts.xml"
with open(output_filename, "w") as f:
    f.write(xml_output)

print(f"Saved MuJoCo contacts to {output_filename}")
