import 'package:flutter/material.dart';
import 'camera_page.dart';

void main() {
  runApp(const SkinAgeApp());
}

class SkinAgeApp extends StatelessWidget {
  const SkinAgeApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '皮肤年龄估算',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: const CameraPage(),
      debugShowCheckedModeBanner: false,
    );
  }
}
