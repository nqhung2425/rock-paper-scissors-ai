import numpy as np
import math

def classify_hand_landmarks(hand_landmarks):
    if hand_landmarks is None or not hasattr(hand_landmarks, 'landmark'):
        return "Invalid"
    
    # Cấu hình ngưỡng góc cho từng nhóm ngón tay
    angle_thresholds = {
        'thumb': {
            'straight_min': 120,
            'bent_max': 90
        },
        'middle_fingers': {  # index, middle, ring
            'straight_min': 160,
            'bent_max': 140
        },
        'pinky': {
            'straight_min': 150,
            'bent_max': 130
        }
    }
    
    # Định nghĩa landmarks cho từng ngón tay
    finger_landmarks = {
        'thumb': [2, 3, 4],
        'index': [5, 6, 7, 8],
        'middle': [9, 10, 11, 12],
        'ring': [13, 14, 15, 16],
        'pinky': [17, 18, 19, 20]
    }
    
    def calculate_angle(p1, p2, p3):
        """Tính góc tại điểm p2"""
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0
        
        cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        angle = math.acos(cos_angle)
        return math.degrees(angle)
    
    def get_finger_thresholds(finger_name):
        """Lấy ngưỡng cho từng nhóm ngón"""
        if finger_name == 'thumb':
            return angle_thresholds['thumb']
        elif finger_name == 'pinky':
            return angle_thresholds['pinky']
        else:
            return angle_thresholds['middle_fingers']
    
    def analyze_finger(finger_name):
        """Phân tích một ngón tay"""
        landmarks = finger_landmarks[finger_name]
        joint_angles = []
        joint_states = []
        
        # Khởi tạo avg_angle mặc định
        avg_angle = 0
        
        # Tính góc các khớp (cần ít nhất 3 điểm)
        if len(landmarks) >= 3:
            for i in range(len(landmarks) - 2):
                p1_idx = landmarks[i]
                p2_idx = landmarks[i + 1]
                p3_idx = landmarks[i + 2]
                
                p1 = (hand_landmarks.landmark[p1_idx].x, hand_landmarks.landmark[p1_idx].y)
                p2 = (hand_landmarks.landmark[p2_idx].x, hand_landmarks.landmark[p2_idx].y)
                p3 = (hand_landmarks.landmark[p3_idx].x, hand_landmarks.landmark[p3_idx].y)
                
                angle = calculate_angle(p1, p2, p3)
                joint_angles.append(angle)
                
                # Phân loại dựa trên ngưỡng riêng biệt
                thresholds = get_finger_thresholds(finger_name)
                
                if angle > thresholds['straight_min']:
                    joint_states.append('straight')
                elif angle < thresholds['bent_max']:
                    joint_states.append('bent')
                else:
                    joint_states.append('neutral')
        
        # Tính góc trung bình
        avg_angle = np.mean(joint_angles) if joint_angles else 0
        
        # Quyết định trạng thái tổng thể
        if not joint_states:
            # Nếu không có joint nào, default là flexed
            overall_state = 'flexed'
        else:
            bent_count = joint_states.count('bent')
            straight_count = joint_states.count('straight')
            
            if straight_count > bent_count:
                overall_state = 'extended'
            elif bent_count > straight_count:
                overall_state = 'flexed'
            else:
                # Dựa trên góc trung bình
                thresholds = get_finger_thresholds(finger_name)
                neutral_threshold = (thresholds['straight_min'] + thresholds['bent_max']) / 2
                overall_state = 'extended' if avg_angle > neutral_threshold else 'flexed'
        
        return overall_state, avg_angle
    
    # Phân tích tất cả các ngón
    finger_results = {}
    finger_states = []
    
    for finger_name in finger_landmarks.keys():
        state, avg_angle = analyze_finger(finger_name)
        finger_results[finger_name] = {'state': state, 'avg_angle': avg_angle}
        finger_states.append(1 if state == 'extended' else 0)
    
    # Logic phân loại cử chỉ Oẳn tù tì
    thumb_avg_angle = finger_results['thumb']['avg_angle']
    other_fingers_extended = sum(finger_states[1:])  # index, middle, ring, pinky
    other_fingers_flexed = 4 - other_fingers_extended
    
    # BAO: Tất cả 5 ngón đều duỗi
    if finger_states == [1, 1, 1, 1, 1]:
        return "paper"
    
    # KÉO: Index+Middle duỗi, Ring+Pinky co, Ngón cái < 170°
    elif (finger_states[1:] == [1, 1, 0, 0] and thumb_avg_angle < 170):
        return "scissors"
    
    # BÚA: 4 ngón khác đều co, Ngón cái < 170°
    elif (other_fingers_flexed == 4 and thumb_avg_angle < 170):
        return "rock"
    else:
        return "Invalid"