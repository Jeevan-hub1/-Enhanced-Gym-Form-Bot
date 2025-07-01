# 🏋️ Enhanced Gym Form Bot

An AI-powered fitness coach that provides real-time form analysis and voice feedback for gym exercises using computer vision and pose estimation.

## ✨ Features

### 🎯 Exercise Support
- **Squats**: Deep knee angle analysis, back posture, knee alignment
- **Push-ups**: Body line analysis, elbow angles, shoulder protection
- **Planks**: Core stability, hip alignment, hold time tracking

### 🧠 Intelligent Analysis
- **Bilateral Form Analysis**: Checks both sides of the body for symmetry
- **Real-time Pose Detection**: Uses MediaPipe for accurate landmark detection
- **Quality Scoring**: 0-100% form quality rating per exercise
- **Smart Rep Counting**: Automatic repetition counting with phase detection

### 🔊 Voice Coaching
- **Intelligent TTS Feedback**: Context-aware voice guidance
- **Motivational Variety**: Multiple encouraging phrases to avoid repetition
- **Throttled Corrections**: Smart timing to prevent feedback overload
- **Offline/Online TTS**: Supports both pyttsx3 and gTTS

### 📊 Performance Tracking
- **Session Statistics**: Total reps, perfect reps, quality scores
- **Streak Tracking**: Monitor consecutive perfect repetitions
- **Form History**: Track improvement over time
- **Exercise-specific Metrics**: Plank hold times, squat depth, etc.

### 🎮 Interactive Controls
- **Live Exercise Switching**: Change exercises mid-session
- **Session Reset**: Start fresh anytime
- **Quality Visualization**: Real-time form feedback overlay

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Webcam
- Speakers/Headphones (for voice feedback)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/enhanced-gym-form-bot.git
cd enhanced-gym-form-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Basic Usage (OpenCV)
```bash
python gym_form_bot.py
```

#### Streamlit Web App
```bash
streamlit run streamlit_app.py
```

### Controls

| Key | Action |
|-----|--------|
| `S` | Switch to Squats |
| `P` | Switch to Push-ups |
| `L` | Switch to Planks |
| `R` | Reset Session |
| `Q` | Quit Application |

## 🏗️ Architecture

### Core Components

```
EnhancedGymFormBot/
├── TTSManager          # Voice feedback system
├── Exercise Analysis   # Form analysis algorithms
├── Pose Detection     # MediaPipe integration
├── Quality Scoring    # Performance metrics
└── Visualization      # Real-time feedback display
```

### Key Classes

- **`TTSManager`**: Handles all text-to-speech functionality with intelligent queuing
- **`EnhancedGymFormBot`**: Main class coordinating pose analysis and feedback
- **Exercise Analyzers**: Specialized form analysis for each exercise type

## 🎯 Exercise Analysis Details

### Squats Analysis
- **Knee Angle Tracking**: Prevents dangerous over-squatting
- **Back Angle Monitoring**: Ensures proper spine alignment
- **Bilateral Symmetry**: Checks for even weight distribution
- **Knee-Toe Alignment**: Prevents knee injury

### Push-ups Analysis
- **Body Line Check**: Maintains straight line from head to heels
- **Elbow Angle Tracking**: Optimal range of motion
- **Depth Control**: Prevents shoulder strain
- **Arm Symmetry**: Balanced muscle engagement

### Planks Analysis
- **Core Engagement**: Hip alignment monitoring
- **Hold Time Tracking**: Endurance measurement
- **Form Maintenance**: Continuous posture checking
- **Progressive Goals**: Milestone encouragement

## 🔧 Configuration

### TTS Settings
```python
# In TTSManager.__init__()
self.voice_speed = 150          # Words per minute
self.voice_type = "offline"     # "offline" or "online"
self.feedback_cooldown = 4.0    # Seconds between feedback
```

### Exercise Thresholds
```python
# Customize in EnhancedGymFormBot.__init__()
self.thresholds = {
    "squats": {
        "knee_angle_down": 95,      # Squat depth
        "knee_dangerous": 60,       # Safety threshold
        "back_angle_min": 70,       # Posture range
        "back_angle_max": 110
    },
    # ... other exercises
}
```

## 📊 Performance Metrics

### Quality Scoring System
- **100-80%**: Excellent form
- **79-60%**: Good form with minor issues
- **59-40%**: Needs improvement
- **Below 40%**: Poor form, high injury risk

### Tracked Statistics
- Total repetitions per exercise
- Perfect repetitions (80%+ quality)
- Current and best streaks
- Average form quality
- Session duration
- Form corrections given

## 🛠️ Development

### Project Structure
```
enhanced-gym-form-bot/
├── gym_form_bot.py          # Main application
├── streamlit_app.py         # Web interface
├── requirements.txt         # Dependencies
├── README.md               # Documentation
├── tests/                  # Unit tests
│   ├── test_form_analysis.py
│   └── test_tts_manager.py
└── assets/                 # Images, sounds
    ├── demo_images/
    └── sound_effects/
```

### Running Tests
```bash
pytest tests/ -v --cov=gym_form_bot
```

### Adding New Exercises

1. **Create Analysis Method**:
```python
def analyze_new_exercise_form(self, landmarks) -> Dict:
    # Implement form analysis logic
    pass
```

2. **Add Thresholds**:
```python
self.thresholds["new_exercise"] = {
    "angle_threshold": 90,
    # ... other parameters
}
```

3. **Update TTS Messages**:
```python
self.feedback_messages["new_exercise"] = {
    "good_form": ["Great technique!"],
    # ... other feedback
}
```

## 🔍 Troubleshooting

### Common Issues

**Camera Not Working**
```bash
# Check camera permissions and try different indices
cap = cv2.VideoCapture(1)  # Try 0, 1, 2...
```

**TTS Not Working**
```bash
# Install system TTS dependencies
# Ubuntu/Debian:
sudo apt-get install espeak espeak-data libespeak1 libespeak-dev

# macOS:
brew install espeak

# Windows: Usually works out of the box
```

**Poor Pose Detection**
- Ensure good lighting
- Wear contrasting colors
- Position yourself fully in frame
- Check camera resolution settings

**Performance Issues**
```python
# Reduce MediaPipe complexity
self.pose = self.mp_pose.Pose(
    model_complexity=0,  # Lower complexity
    min_detection_confidence=0.5
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation for API changes
- Test on multiple platforms

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MediaPipe**: Google's pose estimation framework
- **OpenCV**: Computer vision library
- **pyttsx3**: Text-to-speech conversion
- **Streamlit**: Web app framework

## 🔮 Future Enhancements

- [ ] Additional exercises (deadlifts, lunges, etc.)
- [ ] Machine learning form prediction
- [ ] Mobile app version
- [ ] Workout program integration
- [ ] Social features and leaderboards
- [ ] Advanced biomechanics analysis
- [ ] Custom exercise creation
- [ ] Integration with fitness trackers
om

---

**Made with ❤️ for the fitness community**

*Train smart, train safe, train with AI!* 🏋️‍♀️🤖