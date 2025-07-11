import 'dart:io';
import 'package:chatbot/services/AppService.dart';
import 'package:chatbot/styles.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import '../services/AuthenticationService.dart';

class ClassificationScreen extends StatefulWidget {
  @override
  _ClassificationScreenState createState() => _ClassificationScreenState();
}

class _ClassificationScreenState extends State<ClassificationScreen> {
  XFile? _selectedImage;
  final ImagePicker _picker = ImagePicker();

  List<dynamic>? _typePredictions;
  List<dynamic>? _schoolPredictions;
  bool _isLoading = false;

  String _selectedModel = 'model_semart';

  final List<Map<String, String>> _modelOptions = [
    {'label': 'SEMART (26 schools x 10 types)', 'value': 'model_semart'},
    {'label': 'Balanced (8 schools x 8 types)', 'value': 'model_balanced'},
  ];

  Future<void> _pickImage(ImageSource source) async {
    final pickedImage = await _picker.pickImage(source: source);
    if (pickedImage != null) {
      setState(() {
        _selectedImage = pickedImage;
        _typePredictions = null;
        _schoolPredictions = null;
      });

      if (kDebugMode) {
        print("Image size: ${File(pickedImage.path).lengthSync()} bytes");
      }
    }
  }

  Future<void> _sendRequest() async {
    if (_selectedImage == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Please select an image first!')),
      );
      return;
    }

    setState(() {
      _isLoading = true;
      _typePredictions = null;
      _schoolPredictions = null;
    });

    try {
      File imageFile = File(_selectedImage!.path);
      var result = await AppService().classifyImage(imageFile, _selectedModel);

      setState(() {
        _typePredictions = result['type'];
        _schoolPredictions = result['school'];
      });
    } catch (e) {
      setState(() {
        _typePredictions = [];
        _schoolPredictions = [];
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to classify image: $e')),
      );
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  Widget _buildPredictionSection(String title, List<dynamic>? predictions) {
    if (predictions == null) {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title, style: GoogleFonts.raleway(fontSize: 20, fontWeight: FontWeight.bold)),
          const SizedBox(height: 10),
          Text("Waiting for prediction...", style: GoogleFonts.raleway(fontSize: 16, fontStyle: FontStyle.italic)),
        ],
      );
    }

    // Filter out predictions with null score
    final validPredictions = predictions.where((item) => item['score'] != null).toList();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(title, style: GoogleFonts.raleway(fontSize: 20, fontWeight: FontWeight.bold)),
        const SizedBox(height: 10),
        if (validPredictions.isEmpty)
          Text(
            "The model is unsure. Try another image or model.",
            style: GoogleFonts.raleway(fontSize: 16, fontStyle: FontStyle.italic),
          )
        else
          ...validPredictions.map((item) {
            final double score = (item['score'] as num).toDouble();
            return Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  item['label'],
                  style: GoogleFonts.raleway(fontSize: 16, fontWeight: FontWeight.w500),
                ),
                const SizedBox(height: 4),
                ClipRRect(
                  borderRadius: BorderRadius.circular(4),
                  child: LinearProgressIndicator(
                    value: score / 100.0,
                    backgroundColor: Colors.grey.shade300,
                    valueColor: AlwaysStoppedAnimation<Color>(AppColors.primaryColor),
                    minHeight: 8,
                  ),
                ),
                const SizedBox(height: 12),
              ],
            );
          }).toList(),
      ],
    );
  }


  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.backgroundColor,
      appBar: AppBar(
        title: Text('Classification', style: AppTextStyles.appBar),
        centerTitle: true,
        backgroundColor: AppColors.primaryColor,
        iconTheme: const IconThemeData(color: Colors.white),
        actions: [
          IconButton(
            icon: const Icon(Icons.logout),
            onPressed: () async {
              await AuthenticationService().signout(context: context);
            },
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _selectedImage == null
                ? _buildNoImageSelected()
                : Image.file(
              File(_selectedImage!.path),
              height: 250,
              width: double.infinity,
              fit: BoxFit.cover,
            ),
            const SizedBox(height: 20),
            Row(
              children: [
                Flexible(
                  fit: FlexFit.tight,
                  child: ElevatedButton(
                    style: AppButtonStyles.smallerButtons,
                    onPressed: () => _pickImage(ImageSource.gallery),
                    child: Text("Gallery", style: AppTextStyles.smallButtonStyle),
                  ),
                ),
                const SizedBox(width: 10),
                Flexible(
                  fit: FlexFit.tight,
                  child: ElevatedButton(
                    style: AppButtonStyles.smallerButtons,
                    onPressed: () => _pickImage(ImageSource.camera),
                    child: Text("Camera", style: AppTextStyles.smallButtonStyle),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(10),
              ),
              child: DropdownButtonHideUnderline(
                child: DropdownButton<String>(
                  value: _selectedModel,
                  isExpanded: true,
                  iconEnabledColor: AppColors.buttonColor,
                  dropdownColor: Colors.white,
                  style: GoogleFonts.raleway(
                    color: AppColors.buttonColor,
                    fontWeight: FontWeight.bold,
                    fontSize: 18,
                  ),
                  onChanged: (String? newValue) {
                    if (newValue != null) {
                      setState(() {
                        _selectedModel = newValue;
                      });
                    }
                  },
                  items: _modelOptions.map((model) {
                    return DropdownMenuItem<String>(
                      value: model['value'],
                      child: Text(
                        model['label']!,
                        style: GoogleFonts.raleway(
                          color: AppColors.buttonColor,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    );
                  }).toList(),
                ),
              ),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              style: AppButtonStyles.elevatedButton,
              onPressed: _sendRequest,
              child: Text(
                'Send Request',
                style: GoogleFonts.raleway(fontSize: 18, color: Colors.white, fontWeight: FontWeight.bold),
              ),
            ),
            const SizedBox(height: 40),

            // OUTPUT
            _isLoading
                ? Center(
              child: Column(
                children: [
                  CircularProgressIndicator(),
                  const SizedBox(height: 12),
                  Text("Classifying...", style: GoogleFonts.raleway(fontSize: 16)),
                ],
              ),
            )
                : Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildPredictionSection('Type Prediction', _typePredictions),
                const SizedBox(height: 20),
                _buildPredictionSection('School Prediction', _schoolPredictions),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildNoImageSelected() {
    return Container(
      height: 200,
      width: double.infinity,
      decoration: BoxDecoration(
        color: Colors.grey[200],
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.grey),
      ),
      child: Center(
        child: Text(
          'No image selected',
          style: GoogleFonts.raleway(fontSize: 18, color: Colors.grey[700]!),
        ),
      ),
    );
  }
}
