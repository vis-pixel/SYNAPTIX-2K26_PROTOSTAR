# BioRhythm Fusion Band
**Circadian & Micro-Inflammation Based Early Disease Predictor**
*Engineering Final Year Project Report*

---

## Abstract
Traditional wearable health monitoring systems primarily rely on threshold-based alerts, notifying users only when physiological parameters (e.g., heart rate, SpO₂) cross critical boundaries. However, most diseases—such as viral infections, systemic inflammation, and metabolic imbalances—begin subtly, manifesting as micro-deviations across multiple biological rhythms before acute symptoms occur. This project introduces the **BioRhythm Fusion Band**, a novel real-time wearable system designed to predict early disease onset by identifying "Micro-Deviation Health Signatures." Instead of tracking single-signal thresholds, the proposed architecture employs a multi-signal correlation analysis across heart rate variability (HRV), continuous skin temperature gradients, electrodermal activity (EDA), sleep micro-fragmentation, and sweat biomarkers. Using a Long Short-Term Memory (LSTM) anomaly detection engine paired with Bayesian probabilistic risk scoring, the system detects correlated variations that individually appear normal but collectively indicate pre-symptomatic pathology. 

## 1. Introduction
The advent of wearable biosensors has democratized continuous health monitoring. Commercial smartwatches successfully track fitness and basic cardiovascular health. However, their reliance on generic, population-based thresholds limits their diagnostic capacity to late-stage abnormality detection. The **BioRhythm Fusion Band** addresses this limitation by treating human physiology as an interconnected, multi-variate system. By capturing synchronized micro-deviations—such as a simultaneously slight resting HR elevation, minor HRV drop, and a +0.4°C skin temperature rise—the system can forecast immune responses, overtraining syndromes, and hormonal burnout up to 24–48 hours before overt physical symptoms emerge.

## 2. Literature Review
Current literature in wearable health informatics emphasizes single-modality thresholding:
- **Cardiovascular Monitoring:** High dependency on peak heart rate and irregular ECG limits detection to overt arrhythmias (e.g., Atrial Fibrillation).
- **Temperature & Sweat:** Typically measured in isolation, leading to high false-positive rates due to environmental factors.
- **Sleep Tracking:** Uses accelerometers and basic HRV to track sleep stages but lacks continuous circadian integration during waking hours.
Current AI models in literature often employ simple decision trees or logistic regression on singular time-series, which fail to capture the complex inter-modality dependencies indicative of early pathogenesis.

## 3. Problem Statement
Disease and physiological burnout do not begin at extreme values; they manifest as interconnected, low-magnitude deviations. For example, a heart rate of 82 bpm, a skin temperature increase of 0.4°C, and a slight reduction in REM sleep are all individually classified as "Normal" by conventional wearables. However, the exact synchronization of these micro-deviations forms an early infection signature. The problem addressed in this project is the inability of current wearable architectures to continuously model the *cross-signal interaction* of physiological parameters and establish personalized, dynamic baselines for pre-symptomatic risk prediction.

## 4. Proposed System Architecture

The system utilizes a 4-tier processing architecture covering edge data collection to predictive risk rendering.

```mermaid
flowchart LR
    subgraph Layer 1: Sensor Layer
        PPG[PPG Sensor]
        SKIN[Skin Temp Sensor]
        EDA[EDA / Galvanic Sensor]
        SWEAT[Micro-sweat Sensor]
        IMU[IMU / Accelerometer]
    end

    subgraph Layer 2: Data Processing
        DEN[Signal Denoising]
        FEAT[Feature Extraction]
        BASE[Dynamic Baseline Modeling]
    end

    subgraph Layer 3: AI Engine
        FUSE[Time-Series Fusion]
        LSTM[LSTM Anomaly Detection]
        BAYES[Bayesian Risk Classifier]
    end

    subgraph Layer 4: Smart Alert System
        SCORE[Risk Probability Score]
        HEAT[Correlation Heatmap UI]
    end

    PPG --> DEN
    SKIN --> DEN
    EDA --> DEN
    SWEAT --> DEN
    IMU --> DEN
    
    DEN --> FEAT
    FEAT --> BASE
    BASE --> FUSE
    FUSE --> LSTM
    LSTM --> BAYES
    BAYES --> SCORE
    BAYES --> HEAT
```

## 5. Hardware Components
1. **Photoplethysmography (PPG) Interface:** High-resolution optical sensors (Green/Red/IR LEDs) to capture volumetric variations of peripheral blood circulation, extracting Heart Rate (HR) and Heart Rate Variability (HRV).
2. **NTC Thermistor / Infrared Skin Sensor:** Captures high-fidelity continuous skin temperature gradients ($\Delta T$).
3. **EDA/GSR Electrodes:** Measures Electrodermal Activity, serving as a proxy for sympathetic nervous system arousal and stress hormones.
4. **Electrochemical Micro-Sweat Sensor:** Non-invasive epidermal microfluidic sensor to detect electrolyte imbalances and pH shifts.
5. **6-Axis IMU (Accelerometer & Gyroscope):** Captures physical activity status to contextually filter physiological signals (e.g., ignoring high HR during exercise) and models sleep micro-fragmentation.

## 6. Software & AI Model Design
The mathematical backbone of the BioRhythm AI Engine abandons traditional rule-based logic in favor of Multivariate Time-Series Modeling. 
- **Signal Denoising:** Utilizes Wavelet Transforms to remove motion artifacts from PPG and EDA.
- **Adaptive Baseline:** An Exponential Moving Average (EMA) algorithm combined with circadian sinusoidal fitting maintains a personalized baseline $B_t$ for all signals.
- **LSTM Autoencoder:** An deep sequence model that learns the "normal" multivariate correlation of the individual user over a 14-day initialization period. The model attempts to reconstruct the input vectors; high reconstruction error indicates a multi-signal anomaly.
- **Bayesian Risk Scoring:** Converts the mathematical anomaly vector into a probabilistic medical risk percentage.

## 7. Mathematical Modeling

**7.1. State Vector Definition**
At any timestamp $t$, the physiological state is defined as a multi-dimensional vector $X_t$:
$$ X_t = [HR_t, HRV_t, \Delta T_t, EDA_t, S_t, M_t] $$
Where $S_t$ is sweat chemical shift and $M_t$ is the micro-fragmentation index.

**7.2. Dynamic Deviation Indexing (Mahalanobis Distance)**
Instead of Euclidean limits, we calculate the correlation-aware Mahalanobis Distance ($D_M$) from the user's personalized active mean vector $\mu_t$ and covariance matrix $\Sigma$:
$$ D_M(X_t) = \sqrt{(X_t - \mu_t)^T \Sigma^{-1} (X_t - \mu_t)} $$

**7.3. LSTM Anomaly Detection**
The LSTM updates its hidden state $h_t$ based on historical patterns:
$$ f_t = \sigma(W_f \cdot [h_{t-1}, X_t] + b_f) $$
$$ i_t = \sigma(W_i \cdot [h_{t-1}, X_t] + b_i) $$
$$ \tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, X_t] + b_C) $$
$$ C_t = f_t \ast C_{t-1} + i_t \ast \tilde{C}_t $$
An anomaly score is inversely proportional to the log-likelihood of $X_t$ given $h_{t-1}$.

**7.4. Bayesian Risk Prediction**
To predict disease $D$ given continuous anomaly evidence $E$:
$$ P(D \mid E) = \frac{P(E \mid D) \cdot P(D)}{P(E)} $$

## 8. Algorithm Flowchart

```mermaid
graph TD
    START((Start Setup)) --> COL[Collect 14 Days of Multivariate Data]
    COL --> INIT[Train Patient-Specific LSTM Baseline]
    INIT --> RT[Real-Time Continuous Sampling Window t]
    RT --> EXT[Extract Time & Frequency Features]
    EXT --> CON[Contextual Filter using IMU data]
    CON --> INV{Is User Exercising?}
    INV -- Yes --> LOG[Log as Fitness Data] --> RT
    INV -- No --> NORM[Normalize Vector against Adaptive Baseline]
    NORM --> LSTM[Feed Vector through LSTM Engine]
    LSTM --> ERR[Compute Reconstruction Error e_t]
    ERR --> BAYES[Feed Error to Bayesian Classifier]
    BAYES --> EVAL{P_Disease > Threshold?}
    EVAL -- Yes --> RISK[Activate Early Warning System]
    RISK --> HEAT[Generate Multi-Signal Heatmap]
    EVAL -- No --> UPDATE[Micro-Adjust Adaptive Baseline]
    UPDATE --> RT
    LOG --> RT
```

## 9. Dataset & Training Strategy
- **Pre-Training:** The base LSTM models will be pre-trained on openly available physiological datasets like MIMIC-III (multi-parameter time series) and the WESAD (Wearable Stress and Affect Detection) dataset.
- **Personalized Transfer Learning:** The network uses the first two weeks of user wear to perform few-shot adaptation, locking in circadian variations unique to that individual's chronotype.

## 10. Risk Prediction Framework
Rather than producing a binary "Sick/Healthy" alert, the band generates a **Multi-Signal Correlation Heatmap**. 
- **Y-Axis**: Distinct biological parameters.
- **X-Axis**: Time in hours.
- **Color Gradients**: Represent the variance ($\sigma$) from personalized circadian baselines. 
*Example Signature*: A cluster showing +1.2$\sigma$ in Temp, -1.5$\sigma$ in HRV, +0.8$\sigma$ in EDA during the sleep window directly flags an "Early Viral Replication Phase" or "High Systemic Inflammation" warning 48 hours prior to an actual fever.

## 11. Implementation Plan
- **Phase 1: Hardware Interfacing (Weeks 1-3):** Breadboard integration of PPG, NTC, EDA, and IMU with a low-power microcontroller (e.g., ESP32 or nRF52).
- **Phase 2: Data Pipeline (Weeks 4-6):** Developing real-time Bluetooth Low Energy (BLE) transmission to a smartphone/edge-device and implementing wavelet denoising.
- **Phase 3: AI Model Training (Weeks 7-10):** Training the LSTM autoencoder on Python (TensorFlow/PyTorch) using simulated and open-source datasets.
- **Phase 4: Real-time Integration & UI (Weeks 11-14):** Deploying the Lite version of the AI model on the edge device and finalizing the frontend dashboard.
- **Phase 5: Testing & Validation (Weeks 15-16):** Clinical evaluation simulating stress/fatigue events.

## 12. Advantages Over Existing Systems

| Feature | Existing Wearables | BioRhythm Fusion Band |
|---|---|---|
| **Logic Basis** | Fixed Threshold Alerts | Correlated Micro-Pattern Detection |
| **Analysis Scope** | Single isolated signal | Cross-signal interaction modeling |
| **Baseline Types** | Generic population limits | Personalized adaptive continuous baseline |
| **Detection Stage**| Late-stage, overt abnormality | Sub-clinical, pre-symptomatic deviations |
| **Monitored Scope**| Physical / Cardiovascular | Integrated Biological Rhythm & Immune proxies |

## 13. Applications
1. **Working Professionals:** Predicting burnout due to chronic stress accumulation.
2. **Athletes:** Pre-empting overtraining syndrome and tracking precise recovery windows.
3. **Elderly & Post-Surgery:** Identifying early signs of postoperative infections or general inflammatory decline before they become life-threatening.
4. **Epidemiological Defense:** Serving as an early warning grid for corporate health programs during viral seasons.

## 14. Future Enhancements
- Integration of a non-invasive continuous glucose monitoring (CGM) optical module to track metabolic syndrome.
- On-device Edge AI processing using neuromorphic chips to reduce Bluetooth dependency and save battery.
- Haptic feedback integration providing immediate breathing intervention cues when severe autonomic imbalance is detected.

## 15. Conclusion
The **BioRhythm Fusion Band** redefines wearable biomedical engineering by shifting the paradigm from *reactionary threshold alerting* to *predictive correlation modeling*. By leveraging multi-dimensional physiological signals—coupled with advanced neural architectures and probabilistic risk modeling—this system lays the foundation for true preventative digital medicine. It empowers individuals and healthcare providers to intercept disease at its genesis, offering a highly personalized, deeply integrated safeguard for long-term health.

---
*End of Report*
