// This is a basic Flutter widget test for Skin Age App.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:skin_age_app/main.dart';

void main() {
  testWidgets('Skin Age App smoke test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const SkinAgeApp());

    // Verify that camera page loads
    expect(find.text('拍照识别'), findsOneWidget);
  });
}