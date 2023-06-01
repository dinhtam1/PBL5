import requests
import cv2

# Địa chỉ API nhận biết loại hoa
api_url = "http://192.168.1.19:5000/upload"

# Mở kết nối với camera
camera = cv2.VideoCapture(0)

while True:
    # Đọc khung hình từ camera
    ret, frame = camera.read()
    
    # Hiển thị khung hình
    cv2.imshow("Camera", frame)
    
    # Chờ phím nhấn 'q' để dừng quá trình
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Gửi yêu cầu POST đến API để nhận biết loại hoa
    _, img_encoded = cv2.imencode('.jpg', frame)
    response = requests.post(api_url, files={"image": img_encoded.tobytes()})

    # Xử lý kết quả từ API
    if response.status_code == 200:
        result = response.json()
        flower_name = result["flower_name"]
        accuracy = result["accuracy"]
        if accuracy > 0.5:
            print("Loại hoa được nhận biết: ", flower_name)
            print("Độ chính xác: ", accuracy)
        else:
            print("Không đạt ngưỡng chính xác yêu cầu")
    else:
        print("Lỗi khi gửi yêu cầu API")

# Giải phóng camera và đóng cửa sổ hiển thị
camera.release()
cv2.destroyAllWindows()
