#!/bin/bash

while true
do
    echo "TF_OUTPUT"
    TF_OUTPUT=$(timeout 7 rosrun tf tf_echo world zed2_left_camera_frame) # world zed2_left_camera_frame
    echo "$TF_OUTPUT"

    TRANSLATION=$(echo "$TF_OUTPUT" | grep "Translation" | sed -n 's/.Translation: [([^]])]./\1/p')
    echo "==== PARSED TRANSLATION: $TRANSLATION"
 
    ROTATION=$(echo "$TF_OUTPUT" | grep "Rotation" | sed -n 's/.Quaternion [([^]])]./\1/p')
    echo "==== PARSED ROTATION: $ROTATION"

    # rostopic pub /tf std_msgs/String "data: '$TF_OUTPUT'" --once
    rostopic pub /tf_translation std_msgs/Float64MultiArray "data: [$TRANSLATION]" --once
    rostopic pub /tf_rotation std_msgs/Float64MultiArray "data: [$ROTATION]" --once

    sleep 5
done