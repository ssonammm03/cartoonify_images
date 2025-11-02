from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from .model_loader import CartoonifyModel
import base64
import os
from io import BytesIO

# Load your trained model once
model = CartoonifyModel('cartoonify_app/model/gen_e005.pt')
def homepage(request):
    return render(request, 'cartoonify_app/homepage.html')

# ✅ This is the missing function Django complained about
def index(request):
    return render(request, 'cartoonify_app/index.html')

@csrf_exempt
def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']
        file_path = default_storage.save(uploaded_file.name, ContentFile(uploaded_file.read()))
        full_path = default_storage.path(file_path)

        cartoon_image = model.cartoonify(full_path)

        # Save the cartoonified image
        media_dir = os.path.join(os.getcwd(), 'media')
        os.makedirs(media_dir, exist_ok=True)
        cartoon_filename = 'cartoon_' + uploaded_file.name
        cartoon_path = os.path.join(media_dir, cartoon_filename)
        cartoon_image.save(cartoon_path)

        # Encode to Base64 for frontend display
        buffered = BytesIO()
        cartoon_image.save(buffered, format="JPEG")
        cartoon_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return JsonResponse({
            'status': 'success',
            'cartoon_image': cartoon_base64
        })

    return JsonResponse({'status': 'error', 'message': 'Invalid request'})
