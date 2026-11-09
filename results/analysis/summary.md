# BANKING77 final results (test set, 3,080 examples)

## E1: full fine-tuning vs LoRA r=8 vs head-only (3 seeds)

| Setup | Test acc (mean ± sd) | Test macro-F1 | Per-seed acc | Peak alloc MB | Train s | Examples/s | Checkpoint MB | Trainable params |
|---|---|---|---|---|---|---|---|---|
| full_ft | 93.01 ± 0.15 | 93.00 ± 0.14 | 93.18, 92.95, 92.89 | 2,451 | 265 | 150 | 475.8 | 124,704,845 |
| lora_r8 | 91.56 ± 0.18 | 91.56 ± 0.18 | 91.59, 91.72, 91.36 | 930 | 216 | 184 | 3.7 | 944,717 |
| head_only | 21.41 ± 2.26 | 16.53 ± 1.63 | 20.42, 19.80, 23.99 | 637 | 79 | 507 | 2.5 | 649,805 |

## Paired difference vs full fine-tuning (95% hierarchical bootstrap, 10,000 resamples over seeds and test examples)

Practical limit set in advance: -1 point. Within the limit only if the lower end is above -1.

| Comparison | Metric | Mean difference | 95% interval | Decision |
|---|---|---|---|---|
| lora_r8 - full_ft | accuracy | -1.45 | [-2.13, -0.79] | inconclusive |
| lora_r8 - full_ft | macro_f1 | -1.44 | [-2.14, -0.79] | inconclusive |
| head_only - full_ft | accuracy | -71.60 | [-74.06, -68.80] | worse than the 1-point limit |
| head_only - full_ft | macro_f1 | -76.47 | [-78.43, -74.42] | worse than the 1-point limit |

## E3: LoRA rank (seed 42, same learning rate, alpha = 2r)

| Rank | Test acc | Test macro-F1 | Trainable params | Peak alloc MB | Train s | Checkpoint MB |
|---|---|---|---|---|---|---|
| 4 | 91.59 | 91.60 | 797,261 | 927 | 217 | 3.1 |
| 8 | 91.59 | 91.60 | 944,717 | 929 | 210 | 3.7 |
| 16 | 92.08 | 92.08 | 1,239,629 | 935 | 214 | 4.8 |

One seed per rank, so small differences may be seed noise (E1 seed sd is a useful yardstick).

## Supplementary: head-only at larger learning rates (seed 42; outside the pre-registered grid)

| LR | Test acc | Test macro-F1 |
|---|---|---|
| 0.001 | 55.33 | 52.43 |
| 0.003 | 72.37 | 71.79 |

## Where LoRA and full fine-tuning differ

- Same prediction on 94.31% of test examples (matched seeds).
- Only full FT correct: 2.94%. Only LoRA correct: 1.49%. Both wrong: 5.5% (same wrong answer: 4.24%).

Intents where LoRA loses the most F1 (mean over 3 seeds):

| Intent | Full FT F1 | LoRA F1 | Gap |
|---|---|---|---|
| transfer_timing | 87.3 | 80.8 | -6.5 |
| country_support | 96.4 | 90.1 | -6.3 |
| declined_card_payment | 88.2 | 82.5 | -5.7 |
| virtual_card_not_working | 95.7 | 90.4 | -5.3 |
| get_disposable_virtual_card | 89.5 | 84.3 | -5.2 |
| card_delivery_estimate | 91.3 | 86.4 | -4.9 |
| getting_spare_card | 97.5 | 92.7 | -4.8 |
| card_payment_not_recognised | 91.4 | 86.7 | -4.8 |

Most common mistakes (summed over 3 seeds):

| Setup | True intent | Predicted as | Count |
|---|---|---|---|
| full_ft | balance_not_updated_after_bank_transfer | transfer_timing | 15 |
| full_ft | why_verify_identity | verify_my_identity | 13 |
| full_ft | get_disposable_virtual_card | getting_virtual_card | 13 |
| full_ft | fiat_currency_support | exchange_via_app | 12 |
| full_ft | card_arrival | card_delivery_estimate | 11 |
| lora_r8 | card_arrival | card_delivery_estimate | 18 |
| lora_r8 | declined_transfer | declined_card_payment | 17 |
| lora_r8 | get_disposable_virtual_card | getting_virtual_card | 15 |
| lora_r8 | balance_not_updated_after_bank_transfer | transfer_timing | 15 |
| lora_r8 | card_payment_not_recognised | compromised_card | 14 |
