
template <class T> struct DArray {

	T* data;
	int count;
	int reserved;
	int startSize;

	//

	void init();
	int  getReservedCount(int newCount);
	void resize(int newCount);
	void reserve(int reserveCount);
	void copy(T* d, int n);
	void copy(DArray<T> array);
	void copy(DArray<T>* array);
	void dealloc();
	void freeResize(int n);
	void push(T element);
	void push(T* elements, int n);
	void push(DArray* array);
	void insertMove(T element, int index);
	void insert(T element, int index);
	int  find(T value);
	T*   retrieve(int addedCount);

	bool operator==(DArray<T> array);
	bool operator!=(DArray<T> array);

	void clear();
	T    first();
	T    last();
	bool empty();
	T    pop();
	void pop(int n);
	void remove(int i);
	T&   operator[](int i);
	T&   at(int i);
	T*   operator+(int i);
	T*   atr(int i);
};

template <class T> void DArray<T>::init() {
	*this = {};
	startSize = 100;
}

template <class T> int DArray<T>::getReservedCount(int newCount) {
	if(startSize == 0) startSize = 100;

	int reservedCount = max(startSize, reserved);
	while(reservedCount < newCount) reservedCount *= 2;

	return reservedCount;
}

template <class T> void DArray<T>::resize(int newCount) {
	int reservedCount = getReservedCount(newCount);

	T* newData = mallocArray(T, reservedCount);
	copyArray(newData, data, T, count);

	if(data) free(data);
	data = newData;
	reserved = reservedCount;
}

template <class T> void DArray<T>::reserve(int reserveCount) {
	if(reserveCount > reserved) {
		resize(reserveCount);
	}
}

template <class T> void DArray<T>::copy(T* d, int n) {
	count = 0;
	push(d, n);
}

template <class T> void DArray<T>::copy(DArray<T> array) {
	return copy(array.data, array.count);
}
template <class T> void DArray<T>::copy(DArray<T>* array) {
	return copy(*array);
}

template <class T> void DArray<T>::dealloc() {
	if(data) {
		free(data);
		count = 0;
		data = 0;
		reserved = 0;
	}
}

template <class T> void DArray<T>::freeResize(int n) {
	dealloc();
	resize(n);
}

template <class T> void DArray<T>::push(T element) {
	if(count == reserved) resize(count+1);

	data[count++] = element;
}

template <class T> void DArray<T>::push(T* elements, int n) {
	if(count+n-1 >= reserved) resize(count+n);

	copyArray(data+count, elements, T, n);
	count += n;
}

template <class T> void DArray<T>::push(DArray* array) {
	push(array->data, array->count);
}

template <class T> void DArray<T>::insertMove(T element, int index) {
	if(index > count-1) return push(element);

	if(count == reserved) resize(count+1);

	moveArray(data+index+1, data+index, T, count-(index+1));
	data[index] = element;
	count++;
}

template <class T> void DArray<T>::insert(T element, int index) {
	myAssert(index <= count);
	
	if(index == count) return push(element);
	push(data[index]);
	data[index] = element;
}

template <class T> int DArray<T>::find(T value) {
	for(int i = 0; i < count; i++) {
		if(value == data[i]) return i+1;
	}

	return 0;
}

template <class T> T* DArray<T>::retrieve(int addedCount) {
	if(count+addedCount-1 >= reserved) resize(count+addedCount);

	T* p = data + count;
	count += addedCount;

	return p;
}

template <class T> bool DArray<T>::operator==(DArray<T> array) {
	if(count != array.count) return false;
	for(int i = 0; i < count; i++) {
		if(data[i] != array.data[i]) return false;
	}
	return true;
}
template <class T> bool DArray<T>::operator!=(DArray<T> array) { return !(*this == array); }

template <class T> void DArray<T>::clear()           { count = 0; }
template <class T> T    DArray<T>::first()           { return data[0]; }
template <class T> T    DArray<T>::last()            { return data[count-1]; }
template <class T> bool DArray<T>::empty()           { return count == 0; };
template <class T> T    DArray<T>::pop()             { return data[--count]; }
template <class T> void DArray<T>::pop(int n)        { count -= n; }
template <class T> void DArray<T>::remove(int i)     { data[i] = data[--count]; }
template <class T> T&   DArray<T>::operator[](int i) { return data[i]; }
template <class T> T&   DArray<T>::at(int i)         { return data[i]; }
template <class T> T*   DArray<T>::operator+(int i)  { return data + i; }
template <class T> T*   DArray<T>::atr(int i)        { return data + i; }

//

template <class T> struct HashTable {

	uint* hashArray; // 0 means unused.
	T* data;
	int size;
	int count;
	int maxCount;
	float maxCountPercent;

	uint (*hashFunction) (T* element);

	//

	void init(int size);
	void init(int size, uint (*hashFunction) (T*), float maxCountPercent = 0.8f);
	void add(T* element);
	void resize();
	T* find(uint hash);
	void clear();
	void free();
};

template <class T> void HashTable<T>::init(int size) {
	hashArray = mallocArray(uint, size);
	data = mallocArray(T, size);

	this->size = size;
	count = 0;
	maxCount = size * maxCountPercent;

	clear();
}

template <class T> void HashTable<T>::init(int size, uint (*hashFunction) (T*), float maxCountPercent = 0.8f) {
	this->hashFunction = hashFunction;
	this->maxCountPercent = maxCountPercent;

	init(size);
}

template <class T> void HashTable<T>::add(T* element) {
	if(count == maxCount-1) resize();

	uint hash = hashFunction(element);
	int index = hash % size;
	while(hashArray[index] != 0) index = (index + 1) % size;
	hashArray[index] = hash+1;
	data[index] = *element;

	count++;
}

template <class T> void HashTable<T>::resize() {
	uint* oldHashArray = hashArray;
	T* oldData = data;

	int oldSize = size;
	init(size * 2);

	for(int i = 0; i < oldSize; i++) {
		if(oldHashArray[i] != 0) add(oldData + i);
	}

	::free(oldHashArray);
	::free(oldData);
}

template <class T> T* HashTable<T>::find(uint hash) {
	int index = hash % size;
	hash += 1;
	while(hashArray[index] != hash) index = (index + 1) % size;

	return data + index;
}

template <class T> void HashTable<T>::clear() {
	memset(hashArray, 0, sizeof(uint)*size);
}

template <class T> void HashTable<T>::free() {
	::free(oldHashArray);
	::free(oldData);
	count = 0;
	size = 0;
}

//

template <class T> struct LinkedList {

	struct Node {
		T data;
		Node* prev;
		Node* next;
	};

	Node* head; // list->prev points to last node.
	int count;

	bool singly;
	void* (*alloc) (int size);

	//

	void init(bool singly = false, void* (*alloc) (int) = 0);
	void insert(T element, int index);
	void append(T element);
	void remove(int index);
	void remove();
	void clear();
};

template <class T> void LinkedList<T>::init(bool singly = false, void* (*alloc) (int) = 0) {
	head = 0;
	count = 0;
	this->singly = singly;

	// We assume custom allocated memory is not beeing freed.
	this->alloc = alloc;
}

template <class T> void LinkedList<T>::insert(T element, int index) {
	Node* newNode = alloc ? (Node*) alloc(sizeof(Node)) : 
							(Node*)malloc(sizeof(Node));
	newNode->data = element;

	if(!singly) {
		if(!head) {
			head = newNode;
			head->next = 0;
			head->prev = newNode;

		} else if(index == 0) {
			newNode->prev = head->prev;
			newNode->next = head;
			head->prev = newNode;

			head = newNode;

		} else if(index == count) {
			Node* node = head->prev;

			node->next = newNode;
			newNode->prev = node;
			newNode->next = 0;
			head->prev = newNode; 

		} else {
			Node* prev = head;
			for(int i = 0; i < index-1; i++) prev = prev->next;

			Node* next = prev->next;
			prev->next = newNode;
			next->prev = newNode;
			newNode->prev = prev;
			newNode->next = next;
		}

	} else {
		if(!head) {
			head = newNode;
			head->next = 0;

		} else if(index == 0) {
			newNode->next = head;
			head = newNode;

		} else {
			Node* node = head;
			for(int i = 0; i < index-1; i++) node = node->next;

			Node* next = node->next;
			node->next = newNode;
			newNode->next = next;
		}
	}

	count++;
}

template <class T> void LinkedList<T>::append(T element) { return insert(element, count); }

template <class T> void LinkedList<T>::remove(int index) {
	if(!singly) {
		if(count == 1) {
			if(!alloc) free(head);
			head = 0;

		} else if(index == 0) {
			Node* next = head->next;
			next->prev = head->prev;
			if(!alloc) free(head);
			head = next;

		} else if(index == count-1) {
			Node* node = head->prev->prev;
			if(!alloc) free(node->next);
			node->next = 0;
			head->prev = node;

		} else {
			Node* node = head;
			for(int i = 0; i < index; i++) node = node->next;

			node->prev->next = node->next;
			node->next->prev = node->prev;

			if(!alloc) free(node);
		}

	} else {
		if(count == 1) {
			if(!alloc) free(head);
			head = 0;

		} else if(index == 0) {
			Node* next = head->next;
			if(!alloc) free(head);
			head = next;

		} else {
			Node* node = head;
			for(int i = 0; i < index-1; i++) node = node->next;

			Node* newNext = node->next->next;
			if(!alloc) free(node->next);
			node->next = newNext;
		}
	}

	count--;
}

template <class T> void LinkedList<T>::remove() { return remove(count-1); }

template <class T> void LinkedList<T>::clear() {
	if(!alloc) {
		if(count == 1) {
			free(head);

		} else {
			Node* node = head->next;
			Node* next = node->next;

			while(next != 0) {
				free(node);
				node = next;
				noxt = node->next;
			}

			free(node);
		}
	}

	head = 0;
	count = 0;
}
