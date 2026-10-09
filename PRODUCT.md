# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Plain HTML, CSS, and JavaScript served by a Python standard-library localhost server. Confirmed by the user. No framework or extra package.

## Users

The project owner, testing Hindi speech recognition on a local Windows computer.

## Product Purpose

Provide a quick way to speak into a microphone and inspect or edit the local model's transcription in a textbox.

## Positioning

The interface uses the project's from-scratch local PyTorch model. It does not use hosted transcription or browser speech recognition.

## Operating Context

The test interface runs on localhost in a desktop browser and uses the machine's microphone. The user reviews and edits the result in place.

## Capabilities and Constraints

- Microphone capture and ASR inference stay on the local machine; captured audio is held in memory and is not written to a file.
- The transcript appears in an editable textbox and is not inserted into another application.
- The interface uses the existing ASR checkpoint and frozen tokenizer.
- Existing project rules prohibit pretrained weights, hosted speech APIs, and importing model code or data from another repository.

## Product Principles

- Keep the speak-to-result path short.
- Make the model's output visible and editable.
- Keep speech processing local.
