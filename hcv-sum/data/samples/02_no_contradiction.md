# Retrieval-Augmented Classification of Clinical Notes

## Abstract

We study whether retrieving similar annotated notes improves the classification of clinical documents into diagnostic categories. We propose RACN, a model that retrieves the five most similar labelled notes from a training corpus and conditions a compact transformer classifier on them. On discharge summaries, RACN improves macro-F1 by 4.1 points over a strong fine-tuned baseline. On radiology reports, retrieval provides no measurable benefit.

## Introduction

Clinical note classification supports cohort selection, billing audits and quality reporting. Large language models achieve high accuracy on this task but are expensive to deploy inside hospital networks, where data often cannot leave the premises. Compact models are cheaper to run but struggle with rare diagnostic categories that appear only a handful of times in training data. Retrieval offers a way to give a compact model access to relevant labelled examples at inference time without increasing its parameter count.

## Method

RACN encodes every training note with a frozen sentence encoder and stores the vectors in an approximate nearest-neighbour index. At inference time, the five most similar training notes and their labels are retrieved and concatenated with the input note. A six-layer transformer classifier is then fine-tuned on these augmented inputs. Training RACN takes about 40% longer than training the baseline because every training example must be augmented with retrieved notes. At inference time, however, RACN runs faster than the dense-retrieval baseline because its nearest-neighbour index is quantised.

## Experiments

We evaluate on two datasets: 48,000 discharge summaries and 31,000 radiology reports, both de-identified and drawn from three hospitals. The baseline is the same six-layer classifier fine-tuned without retrieval. We also compare against a dense-retrieval baseline that uses an unquantised index. All models are trained with five random seeds, and we report the mean macro-F1.

## Results

On discharge summaries, RACN reaches a macro-F1 of 71.3 compared with 67.2 for the baseline, an improvement of 4.1 points. The largest gains occur in rare categories with fewer than 50 training examples. On radiology reports, RACN scores 82.0 and the baseline scores 81.9, a difference that is within the variation across random seeds. We attribute this to the short and highly templated structure of radiology reports, which already contain the information needed for classification. Increasing the number of retrieved notes from five to ten reduced macro-F1 on discharge summaries by 0.8 points, because longer inputs were truncated.

## Limitations

All data come from three hospitals in a single country, so the results may not transfer to other health systems. Retrieval can surface notes from the same patient if the index is not filtered by patient identifier, which would inflate scores; we filtered by patient identifier in all experiments. The approach also requires storing the training corpus at inference time, which some data-governance policies do not allow.
