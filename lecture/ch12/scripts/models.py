import timm

def get_model(model_name, img_size, num_classes=4, pretrained=True, device=None):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    if 'vit' in model_name or 'swin' in model_name:
        model.patch_embed.img_size = img_size

    model = model.to(device)
    return model