import 'dart:math' as math;
import 'dart:developer';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';

typedef NudenetInput = List<List<List<Float32List>>>;
const List<int> _shape = [0, 0, 0, 3];

class NudenetTester {
  OrtSessionOptions? _sessionOptions;
  OrtSession? _session;

  NudenetTester() {
    OrtEnv.instance.init();
    OrtEnv.instance.availableProviders().forEach((element) {
      print('onnx provider=$element');
    });
  }

  reset() {}

  release() {
    _sessionOptions?.release();
    _sessionOptions = null;
    _session?.release();
    _session = null;
    OrtEnv.instance.release();
  }

  void initModel() async {
    _sessionOptions = OrtSessionOptions()
      ..setInterOpNumThreads(1)
      ..setIntraOpNumThreads(1)
      ..setSessionGraphOptimizationLevel(GraphOptimizationLevel.ortEnableAll);
    const assetFileName = 'assets/detector.onnx';
    final rawAssetFile = await rootBundle.load(assetFileName);
    final bytes = rawAssetFile.buffer.asUint8List();
    _session = OrtSession.fromBuffer(bytes, _sessionOptions!);
    log("Onnx Model loaded");
  }

  Future<void> inference(NudenetInput data) async {
    final inputOrt = OrtValueTensor.createTensorWithDataList(data, _shape);
    final inputs = {'input': inputOrt};
    final runOptions = OrtRunOptions();
    final outputs = await _session?.runAsync(runOptions, inputs);
    inputOrt.release();
    runOptions.release();
    outputs?.forEach((element) {
      element?.release();
    });
  }

  Future<void> inferenceTest() async {
    final OrtValueTensor inputOrt =
        OrtValueTensor.createTensorWithDataList(createDummyData(), _shape);

    final List<String> inputNames = _session?.inputNames ?? [];
    log("Input Name : $inputNames");
    if (inputNames.isEmpty) return;

    final List<String> outputNames = _session?.outputNames ?? [];
    log("Output Name : $outputNames");
    if (outputNames.isEmpty) return;

    final inputs = {inputNames.first: inputOrt};

    final runOptions = OrtRunOptions();
    final outputs = await _session?.runAsync(runOptions, inputs, outputNames);
    log(outputs.toString());
    inputOrt.release();
    runOptions.release();
    outputs?.forEach((element) {
      element?.release();
    });
  }

  NudenetInput createDummyData() {
    final random = math.Random();

    final Float32List zList =
        Float32List.fromList(List.generate(3, (l) => random.nextDouble()));
    final List<Float32List> yList = List.generate(300, (l) => zList);
    final List<List<Float32List>> xList = List.generate(300, (l) => yList);
    return List.generate(1, (i) => xList);
  }

  // Future<bool> predict(Float32List data, bool concurrent) async {
  //   final inputOrt = OrtValueTensor.createTensorWithDataList(
  //       data, [_batch, _windowSizeSamples]);
  //   final srOrt = OrtValueTensor.createTensorWithData(_sampleRate);
  //   final hOrt = OrtValueTensor.createTensorWithDataList(_hide);
  //   final cOrt = OrtValueTensor.createTensorWithDataList(_cell);
  //   final runOptions = OrtRunOptions();
  //   final inputs = {'input': inputOrt, 'sr': srOrt, 'h': hOrt, 'c': cOrt};
  //   final List<OrtValue?>? outputs;
  //   if (concurrent) {
  //     outputs = await _session?.runAsync(runOptions, inputs);
  //   } else {
  //     outputs = _session?.run(runOptions, inputs);
  //   }
  //   inputOrt.release();
  //   srOrt.release();
  //   hOrt.release();
  //   cOrt.release();
  //   runOptions.release();

  //   /// Output probability & update h,c recursively
  //   final output = (outputs?[0]?.value as List<List<double>>)[0][0];
  //   _hide = (outputs?[1]?.value as List<List<List<double>>>)
  //       .map((e) => e.map((e) => Float32List.fromList(e)).toList())
  //       .toList();
  //   _cell = (outputs?[2]?.value as List<List<List<double>>>)
  //       .map((e) => e.map((e) => Float32List.fromList(e)).toList())
  //       .toList();
  //   outputs?.forEach((element) {
  //     element?.release();
  //   });

  //   /// Push forward sample index
  //   _currentSample += _windowSizeSamples;

  //   /// Reset temp_end when > threshold
  //   if (output >= _threshold && _tempEnd != 0) {
  //     _tempEnd = 0;
  //   }

  //   /// 1) Silence
  //   if ((output < _threshold) && !_triggered) {
  //     print('vad silence: ${_currentSample / _sampleRate}s');
  //   }

  //   /// 2) Speaking
  //   if ((output >= (_threshold - 0.15)) && _triggered) {
  //     print('vad speaking2: ${_currentSample / _sampleRate}s');
  //   }

  //   /// 3) Start
  //   if (output >= _threshold && !_triggered) {
  //     _triggered = true;

  //     /// minus window_size_samples to get precise start time point.
  //     final speechStart =
  //         _currentSample - _windowSizeSamples - _speechPadSamples;
  //     print('vad start: ${speechStart / _sampleRate}s');
  //   }

  //   /// 4) End
  //   if (output < (_threshold - 0.15) && _triggered) {
  //     if (_tempEnd == 0) {
  //       _tempEnd = _currentSample;
  //     }

  //     /// a. silence < min_slience_samples, continue speaking
  //     if (_currentSample - _tempEnd < _minSilenceSamples) {
  //       print('vad speaking4: ${_currentSample / _sampleRate}s');
  //     }

  //     /// b. silence >= min_slience_samples, end speaking
  //     else {
  //       final speechEnd = _tempEnd > 0
  //           ? _tempEnd + _speechPadSamples
  //           : _currentSample + _speechPadSamples;
  //       _tempEnd = 0;
  //       _triggered = false;
  //       print('vad end: ${speechEnd / _sampleRate}s');
  //     }
  //   }
  //   return _triggered;
  // }
}
