from rest_framework import serializers
from .models import ContentItem, ContentCollection

class ContentItemSerializer(serializers.ModelSerializer):
    file_url = serializers.SerializerMethodField()
    processing_progress_display = serializers.SerializerMethodField()

    class Meta:
        model = ContentItem
        fields = '__all__'
        read_only_fields = ('id', 'user', 'file_size', 'mime_type', 
                           'processing_status', 'processing_progress', 
                           'created_at', 'updated_at', 'processed_at')

    def get_file_url(self, obj):
        if obj.file:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.file.url)
        return None

    def get_processing_progress_display(self, obj):
        return f"{obj.processing_progress}%"

class ContentItemCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = ContentItem
        fields = ('title', 'description', 'content_type', 'file', 
                 'text_content', 'tags')

    def create(self, validated_data):
        validated_data['user'] = self.context['request'].user
        
        # Set file information if file is provided
        if 'file' in validated_data and validated_data['file']:
            file_obj = validated_data['file']
            validated_data['file_size'] = file_obj.size
            validated_data['file_name'] = file_obj.name
            validated_data['mime_type'] = getattr(file_obj, 'content_type', '')

        return super().create(validated_data)

class ContentCollectionSerializer(serializers.ModelSerializer):
    content_items = ContentItemSerializer(many=True, read_only=True)
    content_item_count = serializers.SerializerMethodField()

    class Meta:
        model = ContentCollection
        fields = '__all__'
        read_only_fields = ('id', 'user', 'created_at', 'updated_at')

    def get_content_item_count(self, obj):
        return obj.content_items.count()

class ContentCollectionCreateSerializer(serializers.ModelSerializer):
    content_item_ids = serializers.ListField(
        child=serializers.UUIDField(),
        write_only=True,
        required=False
    )

    class Meta:
        model = ContentCollection
        fields = ('name', 'description', 'tags', 'is_public', 'content_item_ids')

    def create(self, validated_data):
        content_item_ids = validated_data.pop('content_item_ids', [])
        validated_data['user'] = self.context['request'].user
        
        collection = super().create(validated_data)
        
        if content_item_ids:
            content_items = ContentItem.objects.filter(
                id__in=content_item_ids,
                user=self.context['request'].user
            )
            collection.content_items.set(content_items)
        
        return collection
