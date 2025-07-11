import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class AppService {
  final String _baseUrl = "url";

  Future<Map<String, dynamic>> classifyImage(File image, String modelName) async {
    try {
      var request = http.MultipartRequest("POST", Uri.parse("$_baseUrl/classify/"));
      request.files.add(await http.MultipartFile.fromPath('file', image.path));
      request.fields['model_name'] = modelName;

      var response = await request.send();

      if (response.statusCode == 200) {
        var responseData = await response.stream.bytesToString();
        var decodedData = jsonDecode(responseData);
        return {
          "type": decodedData['type_prediction'],
          "school": decodedData['school_prediction']
        };
      } else {
        throw Exception("Failed to classify image. Status code: ${response.statusCode}");
      }
    } catch (e) {
      return {
        "type": [],
        "school": [],
      };
    }
  }


  Future<String> sendChatRequest({String? text, String? imagePath}) async {
    try {
      if (text == null && imagePath == null) {
        return "Error: Both text and image cannot be null.";
      }

      var request = http.MultipartRequest("POST", Uri.parse("$_baseUrl/chat/"));

      if (text != null) {
        request.fields['text'] = text;
      }

      if (imagePath != null) {
        request.files.add(await http.MultipartFile.fromPath('image', imagePath));
      }

      var response = await request.send();

      if (response.statusCode == 200) {
        var responseData = await response.stream.bytesToString();
        var decodedData = jsonDecode(responseData);
        return decodedData['response'] ?? "No response from the model.";
      } else {
        throw Exception("Failed to get response. Status code: ${response.statusCode}");
      }
    } catch (e) {
      return "Error: $e";
    }
  }

}
