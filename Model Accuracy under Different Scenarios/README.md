# Model Accuracy under Different Scenarios (RQ3 and RQ3)


## RQ3. Misclassification Rate on Original Model


Generate adversarial test cases for DeepHunter
```
bash generate_deephunter_adv_examples.sh
```

Generate adversarial test cases for Gradient based Attack
```
bash generate_adv_examples_gradient.sh
```

Measure the missclassification rate
```
python missclassification_rate_original_model.py
```

## RQ4. Misclassification Rate on Defended Model


Defend model using deephunter

```
python retrain_robustness.py
```

Defend model using gradient-based attack
```
python run_adv_train.py
```

Generate adversarial test cases for DeepHunter
```
bash generate_deephunter_adv_examples_on_defendel_model.sh
```

Generate adversarial test cases for Gradient based Attacks
```
bash generate_adv_examples_gradient_on_defended_model.sh
```


Measure the missclassification rate
```
python missclassification_rate_defended_model.py
```

