#!/bin/bash

while true
do
    echo "TF_OUTPUT"
    TF_OUTPUT=$(timeout 5 rosrun tf tf_echo panda_link0 left_camera_link_optical) # world zed2_left_camera_frame
    echo "$TF_OUTPUT"

    TRANSLATION=$(echo "$TF_OUTPUT" | grep "Translation" | sed -n 's/.*Translation: \[\([^]]*\)\].*/\1/p')
    echo "==== PARSED TRANSLATION: $TRANSLATION"
 
    ROTATION=$(echo "$TF_OUTPUT" | grep "Rotation" | sed -n 's/.*Quaternion \[\([^]]*\)\].*/\1/p')
    echo "==== PARSED ROTATION: $ROTATION"

    # rostopic pub /tf std_msgs/String "data: '$TF_OUTPUT'" --once
    rostopic pub /tf_translation std_msgs/Float64MultiArray "data: [$TRANSLATION]" --once
    rostopic pub /tf_rotation std_msgs/Float64MultiArray "data: [$ROTATION]" --once

    sleep 5
done



# #!/bin/bash

# while true
# do
#     TF_OUTPUT=$(rosrun tf tf_echo world zed2_left_camera_frame)

#     TRANSLATION=$(echo "$TF_OUTPUT" | grep "Translation" | awk -F'[' '{print $2}' | awk -F']' '{print $1}')
#     TX=$(echo $TRANSLATION | awk '{print $1}')
#     TY=$(echo $TRANSLATION | awk '{print $2}')
#     TZ=$(echo $TRANSLATION | awk '{print $3}')
    
#     ROTATION=$(echo "$TF_OUTPUT" | grep "Rotation" | awk -F'[' '{print $2}' | awk -F']' '{print $1}')
#     RX=$(echo $ROTATION | awk '{print $1}')
#     RY=$(echo $ROTATION | awk '{print $2}')
#     RZ=$(echo $ROTATION | awk '{print $3}')
#     RW=$(echo $ROTATION | awk '{print $4}')

#     # Output the extracted pose information to a .txt file
#     echo "Translation (x, y, z): $TX, $TY, $TZ" > pose_info.txt
#     echo "Rotation (x, y, z, w): $RX, $RY, $RZ, $RW" >> pose_info.txt
    
#     sleep 0.01
# done