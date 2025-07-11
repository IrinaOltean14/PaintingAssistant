import 'dart:convert';
import 'dart:math';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_chat_types/flutter_chat_types.dart' as types;
import 'package:flutter_chat_ui/flutter_chat_ui.dart';
import 'package:image_picker/image_picker.dart';
import '../services/AuthenticationService.dart';
import '../services/AppService.dart';
import '../styles.dart';
import '../widgets/MessageBuilder.dart';

String randomString() {
  final random = Random.secure();
  final values = List<int>.generate(16, (i) => random.nextInt(255));
  return base64UrlEncode(values);
}

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<StatefulWidget> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final List<types.Message> _messages = [];
  final _user = const types.User(id: '82091008-a484-4a89-ae75-a22bf8d6f3ac');

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(
      title: Text('Chat', style: AppTextStyles.appBar),
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
    body: Chat(
      messages: _messages,
      onSendPressed: (types.PartialText message) {
        print("Text input is disabled, this should not be called.");
      },
      onAttachmentPressed: _handleImageSelection,
      user: _user,
      customMessageBuilder: customMessageBuilder,
      inputOptions: InputOptions(
        enabled: false,

      ),
      theme: const DefaultChatTheme(
        primaryColor: AppColors.buttonColor,
        secondaryColor: AppColors.buttonColor,
        backgroundColor: Color(0xCCA69889),
        inputTextColor: Colors.white,
        inputBackgroundColor: AppColors.primaryColor,
        receivedMessageBodyTextStyle: TextStyle(
          color: Colors.white,
          fontSize: 16,
          fontWeight: FontWeight.w500,
          height: 1.5,
        ),
      ),
    ),
  );

  void _addMessage(types.Message message) {
    setState(() {
      _messages.insert(0, message);
    });
  }

  void _handleImageSelection() async {
    final ImageSource? source = await showDialog<ImageSource>(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text('Choose Image Source'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              ListTile(
                leading: const Icon(Icons.camera_alt),
                title: const Text('Take Photo'),
                onTap: () => Navigator.of(context).pop(ImageSource.camera),
              ),
              ListTile(
                leading: const Icon(Icons.photo),
                title: const Text('Choose from Gallery'),
                onTap: () => Navigator.of(context).pop(ImageSource.gallery),
              ),
            ],
          ),
        );
      },
    );

    if (source != null) {
      final result = await ImagePicker().pickImage(
        source: source,
        imageQuality: 70,
        maxWidth: 1440,
      );

      if (result != null) {
        final bytes = await result.readAsBytes();
        final image = await decodeImageFromList(bytes);

        final textController = TextEditingController(
          text: 'Describe the painting in detail, including its main elements, themes, and possible meaning.',
        );

        final caption = await showDialog<String>(
          context: context,
          builder: (BuildContext context) {
            return AlertDialog(
              title: const Text('Add a Caption (Required)'),
              content: TextField(
                controller: textController,
                maxLines: 3,
                decoration: const InputDecoration(
                  border: OutlineInputBorder(),
                ),
              ),
              actions: [
                TextButton(
                  onPressed: () {
                    if (textController.text.isNotEmpty) {
                      Navigator.of(context).pop(textController.text);
                    }
                  },
                  child: const Text('Send'),
                ),
              ],
            );
          },
        );

        if (caption != null) {
          final message = types.CustomMessage(
            author: _user,
            createdAt: DateTime.now().millisecondsSinceEpoch,
            id: randomString(),
            metadata: {
              'text': caption,
              'image': {
                'uri': result.path,
                'width': image.width.toDouble(),
                'height': image.height.toDouble(),
              },
            },
          );

          _addMessage(message);

          // Show "Assistant is typing..." message
          final typingMessage = types.CustomMessage(
            author: const types.User(id: 'assistant'),
            createdAt: DateTime.now().millisecondsSinceEpoch,
            id: randomString(),
            metadata: {'text': "Assistant is typing..."},
          );

          _addMessage(typingMessage);

          try {
            // Send image and caption to backend
            String response = await AppService().sendChatRequest(
              text: caption,
              imagePath: result.path,
            );

            setState(() {
              _messages.removeWhere((msg) => msg.id == typingMessage.id);
            });

            final responseMessage = types.TextMessage(
              author: const types.User(id: 'assistant'),
              createdAt: DateTime.now().millisecondsSinceEpoch,
              id: randomString(),
              text: response,
            );

            _addMessage(responseMessage);
          } catch (e) {
            _showError("Failed to send image: $e");
          }
        }
      }
    }
  }


  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(message)));
  }
}
