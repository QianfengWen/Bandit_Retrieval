from abc import ABC, abstractmethod

def handle_dataset(dataset_name):
    from .datasets import (
        HotelChicagoDataset,
        HotelLondonDataset,
        HotelMontrealDataset,
        HotelNYCDataset,
        PointRecUSDataset,
        RestaurantNorDataset,
        RestaurantPhiDataset,
        TravelDestDataset,
    )

    datasets = {
        "hotel_chicago": HotelChicagoDataset,
        "hotel_london": HotelLondonDataset,
        "hotel_montreal": HotelMontrealDataset,
        "hotel_nyc": HotelNYCDataset,
        "point_rec_us": PointRecUSDataset,
        "restaurant_nor": RestaurantNorDataset,
        "restaurant_phi": RestaurantPhiDataset,
        "travel_dest": TravelDestDataset,
    }

    try:
        return datasets[dataset_name]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Choose from: {', '.join(sorted(datasets))}"
        ) from exc

class Dataloader(ABC):    
    @abstractmethod
    def load_dataset(self):
        pass

    @abstractmethod
    def load_questions(self):
        pass

    @abstractmethod
    def load_passages(self):
        pass

    @abstractmethod
    def create_relevance_map(self):
        pass

    def load_data(self):
        self.dataset = self.load_dataset()
        question_ids, question_texts = self.load_questions()
        passage_ids, passage_texts = self.load_passages()
        relevance_map = self.create_relevance_map()
        
        return question_ids, question_texts, passage_ids, passage_texts, relevance_map

    def visualize_relevance(self, fig_path):
        relevance = [qrel.relevance for qrel in self.dataset.qrels_iter()]
        plt.hist(relevance, bins=len(set(relevance)))
        plt.xlabel('Relevance')
        plt.ylabel('Frequency')
        plt.title('Relevance Distribution')
        plt.savefig(fig_path)
        return
    
    def save_relevance_map(self, data_dir, relevance_map):
        data_path = os.path.join(data_dir, "relevance_map.json")
        with open(data_path, 'w') as json_file:
            json.dump(relevance_map, json_file, indent=4)
        return
    



        
