import cv2
import time
import random
import numpy as np
import mediapipe as mp
from classify_hand_landmarks import classify_hand_landmarks  # Import hàm của bạn

def put_centered_text(img, text, y, font, scale, color, thickness):
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (img.shape[1] - text_w) // 2
    cv2.putText(img, text, (x, y), font, scale, color, thickness)

def get_winner(user_choice, computer_choice):
    if user_choice == "Invalid":
        return "Selection is invalid"
    
    if user_choice == computer_choice:
        return "draw"
    
    win_conditions = {
        ("rock", "scissors"): "User win",
        ("scissors", "paper"): "User win", 
        ("paper", "rock"): "User win"
    }
    
    if (user_choice, computer_choice) in win_conditions:
        return "User win"
    else:
        return "Computer win"

def show_result_screen(user_choice, computer_choice, result, round_num, user_score, computer_score):
    """Hiển thị màn hình kết quả trong 3 giây"""
    result_frame = np.ones((500, 800, 3), dtype=np.uint8) * 255

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Tiêu đề
    put_centered_text(result_frame, f"=== Result round {round_num} ===", 80, font, 1.2, (255, 0, 0), 3)

    # Lựa chọn
    put_centered_text(result_frame, f"User: {user_choice}", 150, font, 1, (0, 100, 0), 2)
    put_centered_text(result_frame, f"Computer: {computer_choice}", 190, font, 1, (0, 0, 200), 2)

    # Kết quả ván này
    result_color = (0, 255, 0) if result == "User win" else (0, 0, 255) if result == "Computer win" else (100, 100, 100)
    put_centered_text(result_frame, f"Result: {result}", 250, font, 1.2, result_color, 3)

    # Tỉ số chung cuộc
    put_centered_text(result_frame, f"Detail: {user_score} - {computer_score}", 310, font, 1.1, (0, 0, 0), 2)

    # Countdown
    for countdown in range(3, 0, -1):
        temp_frame = result_frame.copy()
        put_centered_text(temp_frame, f"Next round: {countdown}s", 400, font, 1, (255, 0, 0), 2)

    cv2.imshow("Rock Paper Scissors", temp_frame)
    cv2.waitKey(3000)

def show_final_result(user_score, computer_score):
    # Hiển thị kết quả chung cuộc và hỏi chơi tiếp
    result_frame = np.ones((400, 800, 3), dtype=np.uint8) * 255

    # Xác định người win chung cuộc
    if user_score > computer_score:
        winner_text = "USER WIN!"
        winner_color = (0, 255, 0)
    elif user_score < computer_score:
        winner_text = "COMPUTER WIN!"
        winner_color = (0, 0, 255)
    else:
        winner_text = "DRAW!"
        winner_color = (100, 100, 100)

    font = cv2.FONT_HERSHEY_SIMPLEX

    put_centered_text(result_frame, "===== FINAL RESULT =====", 80, font, 1.2, (255, 0, 0), 3)
    put_centered_text(result_frame, f"Final Detail: {user_score} - {computer_score}", 150, font, 1.1, (0, 0, 0), 2)
    put_centered_text(result_frame, winner_text, 220, font, 1.5, winner_color, 3)
    put_centered_text(result_frame, "Press any key to play again", 320, font, 1, (0, 0, 0), 2)
    put_centered_text(result_frame, "Press 'q' to exit", 360, font, 0.8, (0, 0, 0), 2)

    cv2.imshow("Rock Paper Scissors", result_frame)

def play_game():
    """Chơi một game hoàn chỉnh (3 ván)"""
    # Khởi tạo MediaPipe
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(0)
    rounds = 3
    user_score = 0
    computer_score = 0
    
    # Vòng lặp 3 ván
    for r in range(rounds):
        print(f"\n--- Ván {r+1} ---")
        user_choice = "Invalid"
        
        start_time = time.time()
        countdown = 3
        
        # Vòng lặp đếm ngược và nhận diện
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # Mirror effect
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)
            
            # Nhận diện tay
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    user_choice = classify_hand_landmarks(hand_landmarks)
            
            # Hiển thị thông tin game
            cv2.putText(frame, f"Round {r+1}/3", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
            
            cv2.putText(frame, f"Detail: {user_score}-{computer_score}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Đếm ngược
            elapsed = int(time.time() - start_time)
            remaining = countdown - elapsed
            if remaining > 0:
                cv2.putText(frame, f"{remaining}", (frame.shape[1]-80, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
            else:
                cv2.putText(frame, "GO!", (frame.shape[1]-100, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                break
            
            # Hiển thị lựa chọn hiện tại
            cv2.putText(frame, f"Select: {user_choice}", (10, frame.shape[0]-50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Rock Paper Scissors", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return False  # Thoát game
        
        # Sau khi hết thời gian → máy chọn
        computer_choice = random.choice(["rock", "scissors", "paper"])
        result = get_winner(user_choice, computer_choice)
        
        print(f"User: {user_choice} | Computer: {computer_choice}")
        print(result)
        
        # Cập nhật điểm
        if result == "User win":
            user_score += 1
        elif result == "Computer win":
            computer_score += 1
        
        # Hiển thị kết quả ván này (3 giây)
        show_result_screen(user_choice, computer_choice, result, r+1, user_score, computer_score)
    
    # Hiển thị kết quả chung cuộc
    show_final_result(user_score, computer_score)
    
    cap.release()
    return True  # Chơi tiếp

def main():
    """Hàm main - vòng lặp game chính"""
    print("🎮 GAME Rock Paper Scissors!")
    
    while True:
        # Chơi một game (3 ván)
        continue_playing = play_game()
        
        if not continue_playing:
            print("\n👋 Thank you!")
            break
        
        # Chờ người chơi quyết định
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            print("\n👋 Thank you!")
            break
        else:
            print("\n🔄 Another game...")
            continue
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()