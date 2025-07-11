import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
// Culori grlobale
class AppColors {
  static const Color backgroundColor = Color(0xCCFBEEDD); // Light Gray
  static const Color primaryColor = Color(0xFF333333); // Charcoal
  static const Color accentColor = Color(0xFFD4AF37); // Gold
  static const Color buttonColor = Color(0xFF0D6F91); // Navy Blue
  static const Color textColor = Color(0xFF322B2B); // Dark Gray
}

class AppTextStyles {
  static final TextStyle headline = GoogleFonts.raleway(
    fontSize: 35,
    fontWeight: FontWeight.bold,
    color: AppColors.textColor,
  );

  static final TextStyle body = GoogleFonts.raleway(
    fontSize: 16,
    color: AppColors.textColor,
  );

  static final TextStyle body2 = GoogleFonts.raleway(
    fontSize: 16,
    color: AppColors.buttonColor,
  );

  static final TextStyle button = GoogleFonts.raleway(
    fontSize: 18,
    fontWeight: FontWeight.w600,
    color: Colors.white,
  );

  static final TextStyle appBar = GoogleFonts.raleway(
    fontSize: 20,
    color: Colors.white,
    fontWeight: FontWeight.bold,
  );

  static final TextStyle smallButtonStyle = GoogleFonts.raleway(
    fontSize: 20,
    color: AppColors.buttonColor,
    fontWeight: FontWeight.bold,
  );
}

class AppButtonStyles {
  static ButtonStyle elevatedButton = ElevatedButton.styleFrom(
    backgroundColor: AppColors.buttonColor,
    shape: RoundedRectangleBorder(
      borderRadius: BorderRadius.circular(14),
    ),
    minimumSize: const Size(double.infinity, 60),
    elevation: 0,
  );

  static ButtonStyle smallerButtons = ElevatedButton.styleFrom(
    backgroundColor: Colors.white,
    shape: RoundedRectangleBorder(
      borderRadius: BorderRadius.circular(14),
    ),
    minimumSize: const Size(double.infinity, 60),
    elevation: 0,
  );
}