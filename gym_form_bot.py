import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import time
import math
from typing import Dict, List, Tuple, Optional
import json
import pygame
import threading
import pyttsx3
from gtts import gTTS
import io
import tempfile
import os
from queue import Queue
import asyncio

class TTSManager:
    """Enhanced Text-to-Speech functionality with intelligent feedback management"""
    
    def __init__(self):
        self.tts_engine = None
        self.tts_queue = Queue()
        self.is_speaking = False
        self.tts_enabled = True
        self.voice_speed = 150
        self.voice_type = "offline"
        self.last_feedback_by_type = {}  # Track last feedback time by type
        
        self.init_offline_tts()
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()
        
        # Enhanced feedback messages with motivational variety
        self.feedback_messages = {
            "squats": {
                "knee_too_deep": ["Don't go too deep, protect your knees", "Easy on the depth, keep knees safe"],
                "back_straight": ["Keep your back straighter", "Straighten that back", "Back up, chest out"],
                "back_lean": ["Don't lean back too much", "Less lean, more upright"],
                "knee_alignment": ["Keep your knees aligned over your toes", "Knees over toes"],
                "good_form": ["Great form! Keep it up", "Perfect squat!", "Excellent technique"],
                "rep_completed": ["Good rep! Well done", "Nice one!", "That's how it's done"]
            },
            "pushups": {
                "body_straight": ["Keep your body in a straight line", "Straighten that body line"],
                "hips_down": ["Don't let your hips sag", "Lift those hips", "Engage your core"],
                "hips_up": ["Don't pike your hips up", "Lower those hips"],
                "too_deep": ["Don't go too low, protect your shoulders", "Not too deep"],
                "hand_position": ["Keep your hands under your shoulders", "Hands under shoulders"],
                "good_form": ["Excellent push-up form", "Perfect form!", "Outstanding"],
                "rep_completed": ["Perfect push-up! Next one", "Great rep!", "Keep going"]
            },
            "planks": {
                "hips_sag": ["Engage your core, don't let hips sag", "Lift those hips up"],
                "hips_high": ["Lower your hips, keep body straight", "Hips down"],
                "legs_straight": ["Straighten your legs", "Extend those legs"],
                "good_form": ["Perfect plank position", "Hold it steady", "Great hold"],
                "hold_milestone": ["Great hold! Keep going", "You're doing amazing", "Stay strong"],
                "form_break": ["Reset your plank position", "Back to position"]
            },
            "general": {
                "exercise_start": ["Let's begin! Focus on your form", "Here we go!", "Time to work"],
                "session_complete": ["Great workout session! Well done", "Fantastic session"],
                "form_reminder": ["Remember to focus on proper form", "Form over speed"],
                "breathing": ["Don't forget to breathe", "Keep breathing steady"]
            }
        }
    
    def init_offline_tts(self):
        """Initialize offline TTS engine with better error handling"""
        try:
            self.tts_engine = pyttsx3.init()
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Prefer female voices for coaching
                for voice in voices:
                    if any(word in voice.name.lower() for word in ['female', 'zira', 'eva']):
                        self.tts_engine.setProperty('voice', voice.id)
                        break
                else:
                    self.tts_engine.setProperty('voice', voices[0].id)
            
            self.tts_engine.setProperty('rate', self.voice_speed)
            self.tts_engine.setProperty('volume', 0.9)
            
        except Exception as e:
            print(f"Error initializing offline TTS: {e}")
            self.tts_engine = None
    
    def speak_async(self, message_type: str, exercise: str = "general", priority: str = "normal"):
        """Enhanced speak function with intelligent message selection"""
        if not self.tts_enabled:
            return
            
        # Check if we should throttle this type of feedback
        current_time = time.time()
        throttle_key = f"{exercise}_{message_type}"
        
        # Different throttle times based on message type
        throttle_times = {
            "good_form": 8.0,
            "rep_completed": 2.0,
            "hold_milestone": 14.0,
            "form_corrections": 3.0
        }
        
        default_throttle = throttle_times.get(message_type, 4.0)
        
        if throttle_key in self.last_feedback_by_type:
            if current_time - self.last_feedback_by_type[throttle_key] < default_throttle:
                return  # Too soon for this type of feedback
        
        # Get varied message
        text = self.get_varied_feedback_message(exercise, message_type)
        if text:
            self.tts_queue.put({
                "text": text, 
                "priority": priority, 
                "timestamp": current_time,
                "type": throttle_key
            })
            self.last_feedback_by_type[throttle_key] = current_time
    
    def get_varied_feedback_message(self, exercise: str, feedback_type: str) -> str:
        """Get varied feedback messages to avoid repetition"""
        messages = self.feedback_messages.get(exercise, {}).get(feedback_type, [])
        if not messages:
            messages = self.feedback_messages["general"].get(feedback_type, [])
        
        if isinstance(messages, list) and messages:
            import random
            return random.choice(messages)
        elif isinstance(messages, str):
            return messages
        return ""
    
    def _tts_worker(self):
        """Enhanced TTS worker with better queue management"""
        while True:
            try:
                if not self.tts_queue.empty():
                    tts_item = self.tts_queue.get()
                    
                    # Skip very old messages
                    if time.time() - tts_item["timestamp"] > 8:
                        continue
                    
                    # Skip if already speaking unless high priority
                    if self.is_speaking and tts_item["priority"] != "high":
                        continue
                    
                    self._speak_text(tts_item["text"])
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"TTS Worker Error: {e}")
                time.sleep(1)
    
    def _speak_text(self, text: str):
        """Speak text with fallback options"""
        try:
            self.is_speaking = True
            if self.voice_type == "offline" and self.tts_engine:
                self._speak_offline(text)
            else:
                self._speak_online(text)
        except Exception as e:
            print(f"TTS Error: {e}")
        finally:
            self.is_speaking = False
    
    def _speak_offline(self, text: str):
        """Offline TTS with timeout protection"""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"Offline TTS Error: {e}")
    
    def _speak_online(self, text: str):
        """Online TTS with pygame"""
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tts.save(tmp_file.name)
                pygame.mixer.music.load(tmp_file.name)
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                
                os.unlink(tmp_file.name)
        except Exception as e:
            print(f"Online TTS Error: {e}")
            if self.tts_engine:
                self._speak_offline(text)


class EnhancedGymFormBot:
    """Enhanced Gym Form Bot with bilateral analysis and improved feedback"""
    
    def __init__(self):
        # Initialize MediaPipe with optimized settings
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1  # Balance between accuracy and performance
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize enhanced TTS
        self.tts_manager = TTSManager()
        
        try:
            pygame.mixer.init()
        except:
            print("Pygame mixer initialization failed")
        
        # Exercise tracking
        self.exercise_counts = {"squats": 0, "pushups": 0, "planks": 0}
        self.exercise_state = {"squats": "up", "pushups": "up", "planks": "hold"}
        self.current_exercise = "squats"
        
        # Enhanced form tracking
        self.form_issues = []
        self.last_feedback_time = 0
        self.feedback_cooldown = 4.0
        self.last_rep_time = 0
        self.rep_feedback_cooldown = 2.0
        
        # Bilateral analysis support
        self.use_bilateral = True
        
        # Motivational system
        self.motivation_counter = 0
        self.last_motivation_time = 0
        self.motivation_interval = 45
        
        # Enhanced thresholds with bilateral support
        self.thresholds = {
            "squats": {
                "knee_angle_down": 95,
                "knee_angle_up": 160,
                "back_angle_min": 70,
                "back_angle_max": 110,
                "knee_dangerous": 60  # Very deep squat warning
            },
            "pushups": {
                "elbow_angle_down": 95,
                "elbow_angle_up": 160,
                "body_line_tolerance": 15,
                "elbow_dangerous": 40  # Too low warning
            },
            "planks": {
                "back_angle_min": 165,
                "back_angle_max": 195,
                "hip_angle_min": 165,
                "hip_angle_max": 195
            }
        }
        
        # Performance tracking
        self.session_stats = {
            "start_time": time.time(),
            "total_reps": 0,
            "form_corrections": 0,
            "exercise_time": 0,
            "perfect_reps": 0,
            "current_streak": 0,
            "best_streak": 0
        }
        
        # Form quality tracking
        self.rep_quality_history = []
    
    def calculate_angle(self, point1: List[float], point2: List[float], point3: List[float]) -> float:
        """Enhanced angle calculation with error handling"""
        try:
            a = np.array(point1[:2])  # Use only x, y coordinates
            b = np.array(point2[:2])
            c = np.array(point3[:2])
            
            ba = a - b
            bc = c - b
            
            # Handle zero vectors
            ba_norm = np.linalg.norm(ba)
            bc_norm = np.linalg.norm(bc)
            
            if ba_norm == 0 or bc_norm == 0:
                return 0
            
            cosine_angle = np.dot(ba, bc) / (ba_norm * bc_norm)
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            angle = np.arccos(cosine_angle)
            
            return np.degrees(angle)
        except Exception as e:
            print(f"Angle calculation error: {e}")
            return 0
    
    def get_landmark_coords(self, landmarks, landmark_id: int) -> List[float]:
        """Get landmark coordinates with visibility check"""
        if landmarks and len(landmarks.landmark) > landmark_id:
            landmark = landmarks.landmark[landmark_id]
            return [landmark.x, landmark.y, landmark.visibility]
        return [0, 0, 0]
    
    def get_bilateral_coords(self, landmarks, left_id: int, right_id: int) -> Tuple[List[float], List[float]]:
        """Get both left and right landmark coordinates"""
        left_coords = self.get_landmark_coords(landmarks, left_id)
        right_coords = self.get_landmark_coords(landmarks, right_id)
        return left_coords, right_coords
    
    def analyze_squat_form(self, landmarks) -> Dict:
        """Enhanced squat analysis with bilateral support"""
        feedback = {
            "issues": [], 
            "good_form": True, 
            "rep_phase": self.exercise_state["squats"],
            "angles": {},
            "quality_score": 100
        }
        
        if not landmarks:
            return feedback
        
        # Get bilateral landmarks
        left_hip, right_hip = self.get_bilateral_coords(landmarks, 
            self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP)
        left_knee, right_knee = self.get_bilateral_coords(landmarks, 
            self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.RIGHT_KNEE)
        left_ankle, right_ankle = self.get_bilateral_coords(landmarks, 
            self.mp_pose.PoseLandmark.LEFT_ANKLE, self.mp_pose.PoseLandmark.RIGHT_ANKLE)
        left_shoulder, right_shoulder = self.get_bilateral_coords(landmarks, 
            self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
        
        # Calculate angles for both sides
        left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
        left_back_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
        right_back_angle = self.calculate_angle(right_shoulder, right_hip, right_knee)
        
        # Average angles for analysis
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
        avg_back_angle = (left_back_angle + right_back_angle) / 2
        
        feedback["angles"] = {
            "knee": avg_knee_angle,
            "back": avg_back_angle,
            "knee_asymmetry": abs(left_knee_angle - right_knee_angle)
        }
        
        # Rep counting with improved logic
        if avg_knee_angle < self.thresholds["squats"]["knee_angle_down"]:
            if self.exercise_state["squats"] == "up":
                self.exercise_state["squats"] = "down"
        elif avg_knee_angle > self.thresholds["squats"]["knee_angle_up"]:
            if self.exercise_state["squats"] == "down":
                self.exercise_state["squats"] = "up"
                self._complete_rep("squats", feedback)
        
        # Enhanced form checks
        quality_deductions = 0
        
        # Dangerous depth check
        if avg_knee_angle < self.thresholds["squats"]["knee_dangerous"]:
            feedback["issues"].append("Too deep - risk of injury!")
            feedback["good_form"] = False
            quality_deductions += 30
            self.tts_manager.speak_async("knee_too_deep", "squats", "high")
        
        # Back angle checks
        if avg_back_angle < self.thresholds["squats"]["back_angle_min"]:
            feedback["issues"].append("Keep your back straighter")
            feedback["good_form"] = False
            quality_deductions += 20
            self.tts_manager.speak_async("back_straight", "squats", "high")
        elif avg_back_angle > self.thresholds["squats"]["back_angle_max"]:
            feedback["issues"].append("Don't lean back too much")
            feedback["good_form"] = False
            quality_deductions += 15
            self.tts_manager.speak_async("back_lean", "squats", "high")
        
        # Asymmetry check
        if feedback["angles"]["knee_asymmetry"] > 15:
            feedback["issues"].append("Uneven squat - balance both sides")
            feedback["good_form"] = False
            quality_deductions += 25
        
        # Knee alignment check (improved)
        left_knee_toe_diff = abs(left_knee[0] - left_ankle[0])
        right_knee_toe_diff = abs(right_knee[0] - right_ankle[0])
        if max(left_knee_toe_diff, right_knee_toe_diff) > 0.08:
            feedback["issues"].append("Keep knees aligned over toes")
            feedback["good_form"] = False
            quality_deductions += 20
            self.tts_manager.speak_async("knee_alignment", "squats", "high")
        
        # Calculate quality score
        feedback["quality_score"] = max(0, 100 - quality_deductions)
        
        # Positive reinforcement for good form
        if feedback["good_form"] and self.exercise_state["squats"] == "down":
            current_time = time.time()
            if current_time - self.last_feedback_time > 8:
                self.tts_manager.speak_async("good_form", "squats", "low")
                self.last_feedback_time = current_time
        
        return feedback
    
    def analyze_pushup_form(self, landmarks) -> Dict:
        """Enhanced push-up analysis"""
        feedback = {
            "issues": [], 
            "good_form": True, 
            "rep_phase": self.exercise_state["pushups"],
            "angles": {},
            "quality_score": 100
        }
        
        if not landmarks:
            return feedback
        
        # Get bilateral landmarks
        left_shoulder, right_shoulder = self.get_bilateral_coords(landmarks,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
        left_elbow, right_elbow = self.get_bilateral_coords(landmarks,
            self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_ELBOW)
        left_wrist, right_wrist = self.get_bilateral_coords(landmarks,
            self.mp_pose.PoseLandmark.LEFT_WRIST, self.mp_pose.PoseLandmark.RIGHT_WRIST)
        left_hip, right_hip = self.get_bilateral_coords(landmarks,
            self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP)
        left_knee, right_knee = self.get_bilateral_coords(landmarks,
            self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.RIGHT_KNEE)
        
        # Calculate angles
        left_elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        left_body_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
        right_body_angle = self.calculate_angle(right_shoulder, right_hip, right_knee)
        
        avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
        avg_body_angle = (left_body_angle + right_body_angle) / 2
        
        feedback["angles"] = {
            "elbow": avg_elbow_angle,
            "body": avg_body_angle,
            "elbow_asymmetry": abs(left_elbow_angle - right_elbow_angle)
        }
        
        # Rep counting
        if avg_elbow_angle < self.thresholds["pushups"]["elbow_angle_down"]:
            if self.exercise_state["pushups"] == "up":
                self.exercise_state["pushups"] = "down"
        elif avg_elbow_angle > self.thresholds["pushups"]["elbow_angle_up"]:
            if self.exercise_state["pushups"] == "down":
                self.exercise_state["pushups"] = "up"
                self._complete_rep("pushups", feedback)
        
        # Form analysis
        quality_deductions = 0
        
        # Body line check
        if abs(avg_body_angle - 180) > self.thresholds["pushups"]["body_line_tolerance"]:
            if avg_body_angle < 165:
                feedback["issues"].append("Don't let your hips sag")
                self.tts_manager.speak_async("hips_down", "pushups", "high")
            else:
                feedback["issues"].append("Don't pike your hips up")
                self.tts_manager.speak_async("hips_up", "pushups", "high")
            feedback["good_form"] = False
            quality_deductions += 25
        
        # Dangerous depth check
        if avg_elbow_angle < self.thresholds["pushups"]["elbow_dangerous"]:
            feedback["issues"].append("Too low - protect your shoulders")
            feedback["good_form"] = False
            quality_deductions += 30
            self.tts_manager.speak_async("too_deep", "pushups", "high")
        
        # Asymmetry check
        if feedback["angles"]["elbow_asymmetry"] > 20:
            feedback["issues"].append("Uneven push-up - balance both arms")
            feedback["good_form"] = False
            quality_deductions += 20
        
        feedback["quality_score"] = max(0, 100 - quality_deductions)
        
        return feedback
    
    def analyze_plank_form(self, landmarks) -> Dict:
        """Enhanced plank analysis with proper hold time tracking"""
        feedback = {
            "issues": [], 
            "good_form": True, 
            "hold_time": 0,
            "angles": {},
            "quality_score": 100
        }
        
        if not landmarks:
            return feedback
        
        # Get bilateral landmarks
        left_shoulder, right_shoulder = self.get_bilateral_coords(landmarks,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
        left_hip, right_hip = self.get_bilateral_coords(landmarks,
            self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP)
        left_knee, right_knee = self.get_bilateral_coords(landmarks,
            self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.RIGHT_KNEE)
        left_ankle, right_ankle = self.get_bilateral_coords(landmarks,
            self.mp_pose.PoseLandmark.LEFT_ANKLE, self.mp_pose.PoseLandmark.RIGHT_ANKLE)
        
        # Calculate angles
        left_back_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
        right_back_angle = self.calculate_angle(right_shoulder, right_hip, right_knee)
        left_hip_angle = self.calculate_angle(left_shoulder, left_hip, left_ankle)
        right_hip_angle = self.calculate_angle(right_shoulder, right_hip, right_ankle)
        
        avg_back_angle = (left_back_angle + right_back_angle) / 2
        avg_hip_angle = (left_hip_angle + right_hip_angle) / 2
        
        feedback["angles"] = {
            "back": avg_back_angle,
            "hip": avg_hip_angle
        }
        
        # Form analysis
        quality_deductions = 0
        
        if avg_back_angle < self.thresholds["planks"]["back_angle_min"]:
            feedback["issues"].append("Don't let your hips sag")
            feedback["good_form"] = False
            quality_deductions += 25
            self.tts_manager.speak_async("hips_sag", "planks", "high")
        elif avg_back_angle > self.thresholds["planks"]["back_angle_max"]:
            feedback["issues"].append("Don't pike your hips up")
            feedback["good_form"] = False
            quality_deductions += 20
            self.tts_manager.speak_async("hips_high", "planks", "high")
        
        if avg_hip_angle < self.thresholds["planks"]["hip_angle_min"]:
            feedback["issues"].append("Straighten your legs")
            feedback["good_form"] = False
            quality_deductions += 15
            self.tts_manager.speak_async("legs_straight", "planks", "high")
        
        feedback["quality_score"] = max(0, 100 - quality_deductions)
        
        # Handle plank timing
        if feedback["good_form"]:
            if not hasattr(self, 'plank_start_time'):
                self.plank_start_time = time.time()
            
            feedback["hold_time"] = time.time() - self.plank_start_time
            
            # Milestone encouragement
            hold_seconds = int(feedback["hold_time"])
            if hold_seconds > 0 and hold_seconds % 15 == 0:
                if not hasattr(self, 'last_milestone') or self.last_milestone != hold_seconds:
                    self.tts_manager.speak_async("hold_milestone", "planks", "normal")
                    self.last_milestone = hold_seconds
        else:
            # Reset plank timer on form break
            if hasattr(self, 'plank_start_time'):
                delattr(self, 'plank_start_time')
            if hasattr(self, 'last_milestone'):
                delattr(self, 'last_milestone')
            self.tts_manager.speak_async("form_break", "planks", "high")
        
        return feedback
    
    def _complete_rep(self, exercise: str, feedback: Dict):
        """Handle rep completion with quality tracking"""
        self.exercise_counts[exercise] += 1
        self.session_stats["total_reps"] += 1
        
        # Track rep quality
        is_perfect_rep = feedback["good_form"] and feedback.get("quality_score", 0) >= 80
        
        if is_perfect_rep:
            self.session_stats["perfect_reps"] += 1
            self.session_stats["current_streak"] += 1
            self.session_stats["best_streak"] = max(
                self.session_stats["best_streak"], 
                self.session_stats["current_streak"]
            )
        else:
            self.session_stats["current_streak"] = 0
        
        # Add to quality history
        self.rep_quality_history.append(feedback.get("quality_score", 0))
        if len(self.rep_quality_history) > 20:  # Keep last 20 reps
            self.rep_quality_history.pop(0)
        
        # Audio feedback
        if time.time() - self.last_rep_time > self.rep_feedback_cooldown:
            if is_perfect_rep:
                self.tts_manager.speak_async("rep_completed", exercise, "normal")
            self.last_rep_time = time.time()
    
    def draw_enhanced_feedback(self, image, landmarks, feedback: Dict):
        """Enhanced drawing with bilateral landmarks and quality indicators"""
        if landmarks:
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                image, landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
        
        height, width = image.shape[:2]
        
        # Quality indicator (top right)
        quality_score = feedback.get("quality_score", 100)
        quality_color = (0, 255, 0) if quality_score >= 80 else (0, 165, 255) if quality_score >= 60 else (0, 0, 255)
        cv2.putText(image, f"Quality: {quality_score:.0f}%", 
                   (width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, quality_color, 2)
        
        # Exercise counter (bottom left)
        rep_text = f"{self.current_exercise.title()}: {self.exercise_counts[self.current_exercise]}"
        cv2.putText(image, rep_text, (10, height - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Form status
        form_status = "Good Form" if feedback["good_form"] else "Check Form"
        color = (0, 255, 0) if feedback["good_form"] else (0, 0, 255)
        cv2.putText(image, form_status, (10, height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Display current issues (top left)
        y_offset = 30
        for issue in feedback.get("issues", []):
            cv2.putText(image, issue, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += 25
        
        # Plank specific - show hold time
        if self.current_exercise == "planks" and "hold_time" in feedback:
            hold_time = feedback["hold_time"]
            time_text = f"Hold: {hold_time:.1f}s"
            cv2.putText(image, time_text, (width - 200, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Current streak indicator
        if self.session_stats["current_streak"] > 0:
            streak_text = f"Streak: {self.session_stats['current_streak']}"
            cv2.putText(image, streak_text, (width - 200, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Angle display for debugging/coaching (optional)
        if "angles" in feedback:
            angles = feedback["angles"]
            angle_y = height - 120
            for angle_name, angle_value in angles.items():
                if isinstance(angle_value, (int, float)):
                    angle_text = f"{angle_name}: {angle_value:.1f}°"
                    cv2.putText(image, angle_text, (10, angle_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    angle_y -= 20
        
        # Session stats (bottom right)
        stats_text = [
            f"Total Reps: {self.session_stats['total_reps']}",
            f"Perfect: {self.session_stats['perfect_reps']}",
            f"Best Streak: {self.session_stats['best_streak']}"
        ]
        
        for i, stat in enumerate(stats_text):
            cv2.putText(image, stat, (width - 200, height - 100 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Exercise phase indicator
        phase_text = feedback.get("rep_phase", "")
        if phase_text:
            phase_color = (0, 255, 255) if phase_text == "down" else (255, 255, 0)
            cv2.putText(image, f"Phase: {phase_text.title()}", (width // 2 - 60, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, phase_color, 2)
        
        return image

    def get_session_summary(self) -> Dict:
        """Get comprehensive session summary"""
        session_duration = time.time() - self.session_stats["start_time"]
        avg_quality = np.mean(self.rep_quality_history) if self.rep_quality_history else 0
        
        return {
            "duration": session_duration,
            "total_reps": self.session_stats["total_reps"],
            "perfect_reps": self.session_stats["perfect_reps"],
            "best_streak": self.session_stats["best_streak"],
            "average_quality": avg_quality,
            "form_corrections": self.session_stats["form_corrections"],
            "exercise_counts": self.exercise_counts.copy()
        }

    def reset_session(self):
        """Reset session statistics"""
        self.session_stats = {
            "start_time": time.time(),
            "total_reps": 0,
            "form_corrections": 0,
            "exercise_time": 0,
            "perfect_reps": 0,
            "current_streak": 0,
            "best_streak": 0
        }
        self.exercise_counts = {"squats": 0, "pushups": 0, "planks": 0}
        self.rep_quality_history = []
        
        # Reset exercise states
        self.exercise_state = {"squats": "up", "pushups": "up", "planks": "hold"}
        
        # Clear plank timing
        if hasattr(self, 'plank_start_time'):
            delattr(self, 'plank_start_time')
        if hasattr(self, 'last_milestone'):
            delattr(self, 'last_milestone')

    def switch_exercise(self, exercise_name: str):
        """Switch to a different exercise"""
        if exercise_name in self.exercise_counts:
            self.current_exercise = exercise_name
            # Reset exercise-specific states
            if exercise_name == "planks":
                if hasattr(self, 'plank_start_time'):
                    delattr(self, 'plank_start_time')
                if hasattr(self, 'last_milestone'):
                    delattr(self, 'last_milestone')
            
            # Announce exercise switch
            self.tts_manager.speak_async("exercise_start", exercise_name, "high")
            return True
        return False

    def analyze_current_exercise(self, landmarks):
        """Analyze form for the currently selected exercise"""
        if self.current_exercise == "squats":
            return self.analyze_squat_form(landmarks)
        elif self.current_exercise == "pushups":
            return self.analyze_pushup_form(landmarks)
        elif self.current_exercise == "planks":
            return self.analyze_plank_form(landmarks)
        else:
            return {"issues": [], "good_form": True, "quality_score": 100}

    def process_frame(self, frame):
        """Main frame processing method"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(rgb_frame)
        
        # Analyze form
        feedback = self.analyze_current_exercise(results.pose_landmarks)
        
        # Draw feedback on frame
        output_frame = self.draw_enhanced_feedback(frame, results.pose_landmarks, feedback)
        
        return output_frame, feedback

    def cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'pose'):
                self.pose.close()
            if hasattr(self, 'tts_manager') and hasattr(self.tts_manager, 'tts_engine'):
                if self.tts_manager.tts_engine:
                    self.tts_manager.tts_engine.stop()
            pygame.mixer.quit()
        except Exception as e:
            print(f"Cleanup error: {e}")

# Example usage function
def main():
    """Main function to run the gym form bot"""
    bot = EnhancedGymFormBot()
    cap = cv2.VideoCapture(0)
    
    print("Gym Form Bot Started!")
    print("Press 'q' to quit")
    print("Press 's' for squats, 'p' for pushups, 'l' for planks")
    print("Press 'r' to reset session")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            output_frame, feedback = bot.process_frame(frame)
            
            # Display frame
            cv2.imshow('Gym Form Bot', output_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                bot.switch_exercise("squats")
            elif key == ord('p'):
                bot.switch_exercise("pushups")
            elif key == ord('l'):
                bot.switch_exercise("planks")
            elif key == ord('r'):
                bot.reset_session()
                print("Session reset!")
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        bot.cleanup()
        
        # Print session summary
        summary = bot.get_session_summary()
        print("\n=== Session Summary ===")
        print(f"Duration: {summary['duration']:.1f} seconds")
        print(f"Total Reps: {summary['total_reps']}")
        print(f"Perfect Reps: {summary['perfect_reps']}")
        print(f"Best Streak: {summary['best_streak']}")
        print(f"Average Quality: {summary['average_quality']:.1f}%")
        print("Exercise Counts:")
        for exercise, count in summary['exercise_counts'].items():
            print(f"  {exercise.title()}: {count}")

if __name__ == "__main__":
    main()