#!/usr/bin/env python3
"""
RoboTech Field Evaluation Neural Network Trainer
=================================================
Trains a neural network to evaluate game states.

Input:  48-dimensional feature vector (ball + 22 players)
Output: probability that our team scores within 300 cycles

This replaces part of the heuristic evaluation in sample_field_evaluator.cpp.

Usage:
  python3 trainer.py <data_dir> [output_dir]

Data dir should contain CSV files from extract_from_logs.py.
"""

import os
import sys
import numpy as np
import pathlib

# Check for Keras/TensorFlow
try:
    import keras
    from keras import layers, models, activations, losses, optimizers, callbacks
    import keras.backend as K
except ImportError:
    print("ERROR: Keras/TensorFlow not installed.")
    print("Install with: pip install tensorflow keras")
    sys.exit(1)


def load_data(data_dir):
    """Load all CSV files from data directory."""
    all_data = []

    for f in os.listdir(data_dir):
        if not f.endswith('.csv'):
            continue
        filepath = os.path.join(data_dir, f)
        print(f"Loading {filepath}...")
        data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        all_data.append(data)

    if not all_data:
        print(f"No CSV files found in {data_dir}")
        sys.exit(1)

    data = np.concatenate(all_data, axis=0)
    print(f"Total samples: {data.shape[0]}")

    # Shuffle
    np.random.shuffle(data)

    # Split features and labels
    X = data[:, :-1]  # all columns except last
    Y = data[:, -1]   # last column (0 or 1)

    # Train/test split (80/20)
    split = int(len(X) * 0.8)
    return X[:split], Y[:split], X[split:], Y[split:]


def build_model(input_dim):
    """
    Build the field evaluation network.
    Architecture: 48 → 64 → 32 → 16 → 1 (sigmoid)

    Smaller than the unmark network because:
    - Input is 48 dims (vs 290)
    - Output is binary (vs 12-class)
    - Faster inference at runtime (called 500x per decision)
    """
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 trainer.py <data_dir> [output_dir]")
        sys.exit(1)

    data_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else './output/'
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    train_X, train_Y, test_X, test_Y = load_data(data_dir)
    print(f"Training: {len(train_X)} samples ({sum(train_Y):.0f} positive)")
    print(f"Testing:  {len(test_X)} samples ({sum(test_Y):.0f} positive)")

    # Handle class imbalance
    pos_count = sum(train_Y)
    neg_count = len(train_Y) - pos_count
    if pos_count > 0 and neg_count > 0:
        class_weight = {0: 1.0, 1: neg_count / pos_count}
        print(f"Class weight: {class_weight}")
    else:
        class_weight = None

    # Build model
    model = build_model(train_X.shape[1])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0005),
        loss=losses.binary_crossentropy,
        metrics=['accuracy']
    )
    model.summary()

    # Callbacks
    my_callbacks = [
        callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, 'best_field_eval.h5'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]

    # Train
    history = model.fit(
        train_X, train_Y,
        epochs=100,
        batch_size=64,
        class_weight=class_weight,
        validation_data=(test_X, test_Y),
        callbacks=my_callbacks
    )

    # Save final model
    model.save(os.path.join(output_dir, 'field_eval_model.h5'))

    # Evaluate
    test_loss, test_acc = model.evaluate(test_X, test_Y)
    print(f"\nFinal test accuracy: {test_acc:.4f}")
    print(f"Final test loss: {test_loss:.4f}")

    # Save results
    with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
        f.write(f"Test accuracy: {test_acc:.4f}\n")
        f.write(f"Test loss: {test_loss:.4f}\n")
        f.write(f"Train samples: {len(train_X)}\n")
        f.write(f"Test samples: {len(test_X)}\n")

    print(f"\nModel saved to {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Convert model:  python3 ../../tools/CppDNN/script/DecodeKerasModel.py {output_dir}/best_field_eval.h5")
    print(f"  2. Copy weights:   cp best_field_eval.txt ../../build/bin/field_eval_weights.txt")
    print(f"  3. Rebuild team:   cd ../../build && make -j$(nproc)")


if __name__ == '__main__':
    main()
