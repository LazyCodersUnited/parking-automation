import cv2
import numpy as np
def detect_license_plate(image_path, debug=True):
    """
    Detects and crops the license plate from an image using a classical CV pipeline.
    Args:
        image_path (str): Path to the input image.
        debug (bool): If True, shows intermediate steps for tuning.
    Returns:
        plate_crop (numpy.ndarray or None): Cropped license plate image, or None if not found.
    """
    # 1️⃣ Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 2️⃣ Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, d=11, sigmaColor=17, sigmaSpace=17)

    # 3️⃣ Edge detection
    edged = cv2.Canny(blurred, 30, 200)

    # 4️⃣ Find contours
    # contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    #
    # plate_contour = None

    # 5️⃣ Filter contours by shape (quadrilateral)
    # for c in contours:
    #     peri = cv2.arcLength(c, True)
    #     approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    #     if len(approx) == 4:
    #         plate_contour = approx
    #         break
    #
    # if plate_contour is None:
    #     if debug:
    #         print("No plate contour found.")
    #     return None

    # 6️⃣ Create mask & crop plate
    # mask = np.zeros(gray.shape, np.uint8)
    # cv2.drawContours(mask, [plate_contour], 0, 255, -1)
    # x, y, w, h = cv2.boundingRect(plate_contour)
    # plate_crop = img[y:y + h, x:x + w]

    # if debug:
    #     cv2.imshow("Original", img)
    #     cv2.imshow("Gray", gray)
    #     cv2.imshow("Blurred", blurred)
    #     cv2.imshow("Edged", edged)
    #     cv2.imshow("Plate", plate_crop)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #
    # return plate_crop


if __name__ == "__main__":
    # Example usage
    test_image = r"C:\Users\AKSHAT\Desktop\hemant\WhatsApp Image 2024-07-02 at 16.11.37_838626b6.jpg"
    plate = detect_license_plate(test_image, debug=True)
    if plate is not None:
        cv2.imwrite("detected_plate.jpg", plate)
        print("Plate saved as detected_plate.jpg")
    else:
        print("No plate detected.")
