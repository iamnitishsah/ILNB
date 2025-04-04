from rest_framework import viewsets, generics
from rest_framework.permissions import IsAuthenticated
from .models import TradeActivity
from .serializers import TradeActivitySerializer

class TradeActivityViewSet(viewsets.ModelViewSet):
    queryset = TradeActivity.objects.all().order_by('-timestamp')
    serializer_class = TradeActivitySerializer
    permission_classes = [IsAuthenticated]

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class TradeViewSet(generics.ListAPIView):
    serializer_class = TradeActivitySerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return TradeActivity.objects.all()
