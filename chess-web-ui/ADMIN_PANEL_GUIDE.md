# 🎓 Admin Panel - Model Training Guide

## ⚙️ Overview

The Admin Panel allows you to train the Chess AI model using different strategies:
- **Mirror Match** - Two identical models compete against each other
- **Stockfish Trainer** - Model learns by playing against Stockfish engine
- **Archive Alpha** - Alpha-zero style self-play training

---

## 🚀 How to Access

1. Login to Chess Engine at http://localhost:3000
2. Click **"⚙️ Admin"** in the navbar (top right)
3. You'll see the Admin Panel with training controls

---

## 🎯 Training Configuration

### Step 1: Select Training Strategy
Choose one of three available strategies:

| Strategy | Best For | Epochs |
|----------|----------|--------|
| 🔄 **Mirror Match** | Balanced improvement, avoid overfitting | 100 |
| ♞ **Stockfish Trainer** | Learning tactical patterns | 50 |
| 🎓 **Archive Alpha** | Maximum strength (resource intensive) | 200 |

### Step 2: Set Number of Epochs
- Each epoch = one training round
- More epochs = better model but longer training time
- Recommended values vary by strategy (shown in description)

### Step 3: Start Training
- Click **"▶️ Start Training"** button
- Cannot change strategy while training is running
- Monitor progress in real-time on the right panel

---

## 📊 Training Status Panel

### Real-Time Metrics

During training, you'll see:

| Metric | Meaning |
|--------|---------|
| **Status Badge** | 🔴 TRAINING or ⚪ IDLE |
| **Current Status** | Detailed training progress message |
| **Progress Bar** | Visual representation of training progress |
| **Games Played** | Total games completed in training |
| **Loss Value** | Training loss (lower is better) |
| **White Wins** | Games won as white pieces |
| **Black Wins** | Games won as black pieces |
| **Draws** | Number of drawn games |
| **Strategy** | Current training strategy |

### Progress Visualization
- **Progress Bar** shows % completion
- **Epoch Counter** shows current/total epochs
- **Real-time updates** every second

---

## 🛑 Stopping Training

Click **"⏹️ Stop Training"** to:
- Stop training immediately
- Save current model state
- Return to IDLE status

---

## 📚 Understanding Training Strategies

### 🔄 Mirror Match
**How it works:**
- Two identical neural networks play against each other
- Each game generates training data
- Both models learn simultaneously

**Advantages:**
- Balanced difficulty progression
- Learns diverse chess concepts
- Reduces overfitting

**Use Case:**
- General purpose AI improvement
- Best for first-time training

**Duration:** Fast (few seconds per epoch)

---

### ♞ Stockfish Trainer
**How it works:**
- AI model plays against Stockfish engine
- Model learns from playing a strong, proven opponent
- Focuses on defensive and tactical patterns

**Advantages:**
- Learn from one of the world's best engines
- Strong tactical training
- Clear performance metrics

**Use Case:**
- Improve tactical skills
- Learn winning strategies
- Post-training refinement

**Duration:** Medium (needs Stockfish installed)

---

### 🎓 Archive Alpha
**How it works:**
- AlphaZero-style self-play learning
- Deep neural network continuously improves
- Creates new game variations to learn from

**Advantages:**
- Most powerful training method
- Discovers novel strategies
- Achieves superhuman performance

**Use Case:**
- Long-term model improvement
- Research and experimentation
- Maximum AI strength

**Duration:** Slow (resource intensive)

---

## 💡 Training Tips

### For Beginners
1. Start with **Mirror Match** strategy
2. Set epochs to recommended value (100)
3. Monitor progress for first 10-20 epochs
4. Stop if loss value stops improving

### For Advanced Users
1. Use **Stockfish Trainer** for tactical improvement
2. Run **Archive Alpha** overnight for long training
3. Compare results across different strategies
4. Adjust epoch counts based on loss curves

### Performance Optimization
- Train during off-peak hours
- Use **Mirror Match** for quick iterations
- Monitor loss value to avoid overfitting
- Save model after successful training

---

## 📈 Interpreting Loss Value

| Loss Value | Interpretation |
|-----------|-----------------|
| **< 0.1** | Excellent - Model learning well |
| **0.1 - 0.5** | Good - Normal training progress |
| **0.5 - 1.0** | Acceptable - Continue training |
| **> 1.0** | Poor - Training may be ineffective |
| **Increasing** | Overfitting - Consider stopping |

---

## ⚠️ Important Notes

1. **Training is CPU/GPU intensive** - May impact game performance
2. **Model saves automatically** at each epoch
3. **Cannot interrupt mid-epoch** - Wait for next status update
4. **Strategy is locked during training** - Must stop to change
5. **Epochs are approximate** - Actual training time varies

---

## 🔧 Troubleshooting

### Training Won't Start
- Check that AI engine is running (port 5050)
- Ensure you're logged in
- Try refreshing the page

### Training Too Slow
- Use **Mirror Match** instead of Archive Alpha
- Reduce epoch count
- Check system resources

### Loss Value Not Decreasing
- Switch to different strategy
- Increase learning rate (requires code change)
- Reduce number of epochs

### Model Not Loading
- Ensure `danibot.pth` exists in ai-engine folder
- Check Flask app is running properly

---

## 📊 Recommended Training Plans

### Quick Test (5 minutes)
```
Strategy: Mirror Match
Epochs: 10
Expected: Loss should drop from ~2.0 to ~1.5
```

### Standard Training (1-2 hours)
```
Strategy: Mirror Match
Epochs: 100
Expected: Loss should drop from ~2.0 to ~0.2
```

### Intensive Training (4+ hours)
```
Strategy: Archive Alpha
Epochs: 200
Expected: Loss should drop from ~3.0 to ~0.1
```

---

## 🎯 Next Steps After Training

1. **Test the Model**
   - Play against newly trained AI
   - Notice improvement in gameplay

2. **Compare Results**
   - Train with different strategies
   - Compare final loss values

3. **Backup Your Model**
   - Copy `danibot.pth` to safe location
   - Track training results over time

4. **Iterate**
   - Use lessons learned for next training session
   - Experiment with different epoch counts

---

## 📞 Support

For issues with training:
1. Check Flask app logs
2. Verify Stockfish is installed (for Stockfish Trainer)
3. Monitor system resources
4. Check browser console for errors

---

**Happy Training! 🚀**

