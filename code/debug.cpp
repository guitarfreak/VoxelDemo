

#define CONSOLE_SMALL_PERCENT 0.3f
#define CONSOLE_BIG_PERCENT 0.8f

struct Console {
	float pos;
	float targetPos;
	int mode;
	bool isActive;

	float scrollMode;
	float scrollPercent;
	float lastDiff;
	float mouseAnchor;

	char* mainBuffer[256];
	int mainBufferSize;

	char inputBuffer[256];
	// int inputBufferSize;

	int cursorIndex;
	int markerIndex;
	bool mouseSelectMode;
	float cursorTime;

	// char* bodySelectedText;
	int bodySelectionIndex;
	int bodySelectionMarker1, bodySelectionMarker2;
	bool bodySelectionMode;
	bool mousePressedInside;

	/*
		TODO's:
		* Ctrl backspace, Ctrl Delete.
		* Ctrl Left/Right not jumping multiple spaces.
		* Selection Left/Right not working propberly.
		* Ctrl + a.
		* Clipboard.
		* Mouse selection.
		* Mouse release.
		* Cleanup.
		* Scrollbar.
		* Line wrap.

		* Select inside console output.
		  - Clean this up once it's solid.
		- Command history.
		- Command hint on tab.
	*/

	void init(float windowHeight) {
		float smallPos = -windowHeight * CONSOLE_SMALL_PERCENT;
		pos = 0;
		mode = 1;
		// mode = 0;
		targetPos = smallPos;
		cursorIndex = 0;
		markerIndex = 0;
		scrollPercent = 1;
		bodySelectionIndex = -1;


		pushToMainBuffer("This is a test String!");
		pushToMainBuffer("123.456");

		pushToMainBuffer("Nothing!");
		pushToMainBuffer("");

		pushToMainBuffer("This is a test String This is a test StringThis is a test StringThis is a test StringThis is a test StringThis is a test StringThis is a test StringThis is a test StringThis is a test StringThis is a test StringThis is a test StringThis is a test StringThis is a test StringThis is a test StringThis is a test StringThis is a test StringThis is a test StringThis is a test StringThis is a test StringThis is a test StringThis is a test String");

		pushToMainBuffer("Lets get ready to rumbleee Lets get ready to rumbleeeLets get ready to rumbleeeLets get ready to rumbleee Lets get ready to rumbleeeLets get ready to rumbleeeLets get ready to rumbleee    Lets get ready to rumbleee Lets get ready to rumbleeeLets get ready to rumbleeeLets get ready to rumbleeeLets get ready to rumbleeeLets get ready to rumbleeeLets get ready to rumbleeeLets get ready to rumbleeeLets get ready to rumbleee");


		pushToMainBuffer("We were somewhere around Barstow on the edge of the desert when the drugs began to take hold. I remember saying something like “I feel a bit lightheaded; maybe you should drive…” And suddenly there was a terrible roar all around us and the sky was full of what looked like huge bats, all swooping and screeching and diving around the car, which was going about a hundred miles an hour with the top down to Las Vegas. And a voice was screaming: “Holy Jesus! What are these goddamn animals?” Then it was quiet again. My attorney had taken his shirt off and was pouring beer on his chest, to facilitate the tanning process. “What the hell are you yelling about?” he muttered, staring up at the sun with his eyes closed and covered with wraparound Spanish sunglasses. “Never mind,” I said. “It’s your turn to drive.” I hit the brakes and aimed the Great Red Shark toward the shoulder of the highway. No point mentioning those bats, I thought. The poor bastard will see them soon enough. It was almost noon, and we still had more than a hundred miles to go. They would be tough miles. Very soon, I knew, we would both be completely twisted. But there was no going back, and no time to rest. We would have to ride it out. Press-registration for the fabulous Mint 400 was already underway, and we had to get there by four to claim our sound-proof suite. A fashionable sporting-magazine in New York had taken care of the reservations, along with this huge red Chevy convertible we’d just rented off a lot on the Sunset Strip… and I was, after all, a professional journalist; so I had an obligation to cover the story, for good or ill. The sporting editors had also given me $300 in cash, most of which was already spent on extremely dangerous drugs. The trunk of the car looked like a mobile police narcotics lab. We had two bags of grass, seventy-five pellets of mescaline, five sheets of high-powered blotter acid, a salt shaker half full of cocaine, and a whole galaxy of multi-colored uppers, downers, screamers, laughers and also a quart of tequila, a quart of rum, a case of Budweiser, a pint of raw ether and two dozen amyls. All this had been rounded up the night before, in a frenzy of highspeed driving all over Los Angeles County – from Topanga to Watts, we picked up everything we could get our hands on. Not that we needed all that for the trip, but once you get locked into a serious drug-collection, the tendency is to push it as far as you can. The only thing that really worried me was the ether. There is nothing in the world more helpless and irresponsible and depraved than a man in the depths of an ether binge. And I knew we’d get into that rotten stuff pretty soon. Probably at the next gas station. We had sampled almost everything else, and now – yes, it was time for a long snort of ether. And then do the next hundred miles in a horrible, slobbering sort of spastic stupor. The only way to keep alert on ether is to do up a lot of amyls – not all at once, but steadily, just enough to maintain the focus at ninety miles an hour through Barstow. “Man, this is the way to travel,” said my attorney. He leaned over to turn the volume up on the radio, humming along withthe rhythm section and kind of moaning the words: “One toke over the line, Sweet Jesus… One toke over the line...” One toke? You poor fool! Wait till you see those goddamn bats. I could barely hear the radio… slumped over on the far side of the seat, grappling with a tape recorder turned all the way up on “Sympathy for the Devil.” That was the only tape we had, so we played it constantly, over and over, as a kind of demented counterpoint  to the radio. And also to maintain our rhythm on the road. A constant speed is good for gas mileage – and for some reason that seemed important at the time. Indeed. On a trip like this one must be careful about gas consumption. Avoid those quick bursts of acceleration that drag blood to the back of the brain. My attorney saw the hitchhiker long before I did. “Let’s give this boy a lift,” he said, and before I could mount any argument he was stopped and this poor Okie kid was running up to the car with a big grin on his face, saying, “Hot damn! I never rode in a convertible before!” “Is that right?” I said. “Well, I guess you’re about ready, eh?” The kid nodded eagerly as we roared off. “We’re your friends,” said my attorney. “We’re not like the others.” O Christ, I thought, he’s gone around the bend. “No more of that talk,” I said sharply. “Or I’ll put the leeches on you.” He grinned, seeming to understand. Luckily, the noise in the car was so awful – between the wind and the radio and the tape machine – that the kid in the back seat couldn’t hear a word we were saying. Or could he? How long can we maintain? I wondered. How long before one of us starts raving and jabbering at this boy? What will he think then? This same lonely desert was the last known home of the Manson family. Will he make that grim connection when my attorney starts screaming about bats and huge manta rays coming down on the car? If so – well, we’ll just have to cut his head off and bury him somewhere. Because it goes without saying that we can’t turn him loose. He’ll report us at once to some kind of outback nazi law enforcement agency, and they’ll run us down like dogs. Jesus! Did I say that? Or just think it? Was I talking? Did they hear me? I glanced over at my attorney, but he seemed oblivious – watching the road, driving our Great Red Shark along at a hundred and ten or so. There was no sound from the back seat. Maybe I’d better have a chat with this boy, I thought. Perhaps if I explain things, he’ll rest easy. Of course. I leaned around in the seat and gave him a fine big smile… admiring the shape of his skull. “By the way,” I said. “There’s one thing you should probably understand.” He stared at me, not blinking. Was he gritting his teeth? “Can you hear me?” I yelled. He nodded. “That’s good,” I said. “Because I want you to know that we’re on our way to Las Vegas to find the American Dream.” I smiled. “That’s why we rented this car. It was the only way to do it. Can you grasp that?” He nodded again, but his eyes were nervous. “I want you to have all the background,” I said. “Because this is a very ominous assignment – with overtones of extreme personal danger… Hell, I forgot all about this beer; you want one?” He shook his head. “How about some ether?” I said. “What?” “Never mind. Let’s get right to the heart of this thing. You see, about twenty-four hours ago we were sitting in the Polo Lounge of the Beverly Hills Hotel – in the patio section, of course – and we were just sitting there under a palm tree when this uniformed dwarf came up to me with a pink telephone and said, ‘This must be the call you’ve been waiting for all this time, sir.’” I laughed and ripped open a beer can that foamed all over the back seat while I kept talking. “And you know? He was right! I’d been expecting that call, but I didn’t know who it would come from. Do you follow me?” The boy’s face was a mask of pure fear and bewilderment. I blundered on: “I want you to understand that this man at the wheel is my attorney! He’s not just some dingbat I found on the Strip. Shit, look at him! He doesn’t look like you or me, right? That’s because he’s a foreigner. I think he’s probably Samoan. But it doesn’t matter, does it? Are you prejudiced?” “Oh, hell no!” he blurted. “I didn’t think so,” I said. “Because in spite of his race, this man is extremely valuable to me.” I glanced over at my attorney, but his mind was somewhere else. I whacked the back of the driver’s seat with my fist. “This is important, goddamn it! This is  a true story!” The car swerved sickeningly, then straightened out. “Keep your hands off my fucking neck!” my attorney screamed. The kid in the back looked like he was ready to jump right out of the car and take his chances. Our vibrations were getting nasty – but why? I was puzzled, frustrated. Was there no communication in this car? Had we deteriorated to the level of dumb beasts? Because my story was true. I was certain of that. And it was extremely important, I felt, for the meaning of our journey to be made absolutely clear. We had actually been sitting there in the Polo Lounge – for many hours – drinking Singapore Slings with mescal on the side and beer chasers. And when the call came, I was ready. The Dwark approached our table cautiously, as I recall, and when he handed me the pink telephone I said nothing, merely listened. And then I hung up, turning to face my attorney. “That was headquarters,” I said. “They want me to go to Las Vegas at once, and make contact with a Portuguese photographer named Lacerda. He’ll have the details. All I have to do is check into my suite and he’ll seek me out.” My attorney said nothing for a moment, then he suddenly came alive in his chair. “God hell!” he exclaimed. “I think I see the pattern. This one sounds like real trouble!” He tucked his khaki undershirt into his white rayon bellbottoms and called for more drink. “You’re going to need plenty of legal advice before this thing is over,” he said. “And my first advice is that you should rent a very fast car with no top and get the hell out of L.A. for at least forty-eight hours.” He shook his head sadly. “This blows my weekend, because naturally I’ll have to go with you – and we’ll have to arn ourselves.” “Why not?” I said. “If a thing like this is worth doing at all, it’s worth doing right. We’ll need some decent equipment and plenty of cash on the line – if only for drugs and a supersensitive tape recorder, for the sake of a permanent record.” “What kind of a story is this?” he asked. “The Mint 400,” I said. “It’s the richest off-the-road race for motorcycles and dune-buggies in the history of organized sport – a fantastic spectacle in honor of some fatback grossero named Del Webb, who owns the luxurious Mint Hotel in the heart of downtown Las Vegas… at least that’s what the press release says; my man in New York just read it to me.” “Well,” he said, “as your attorney I advise you to buy a motorcycle. How else can you cover a thing like this righteously?” “No way,” I said. “Where can we get hold of a Vincent Black Shadow?” “What’s that?” “A fantastic bike,” I said. “The new model is something like two thousand cubic inches, developing two hundred brake-horsepower at four thousand revolutions per minute on a magnesium frame with two styrofoam seats and a total curb weight of exactly two hundred pounds.” “That sounds about right for this gig,” he said. “It is” I assured him. “The fucker’s not much for turning, but it’s pure hell on the straightaway. It’ll outrun the F-111 until takeoff.” “Takeoff?” he said. “Can we handle that much torque?” “Absolutely,” I said. “I’ll call New York for some cash.”");


		pushToMainBuffer("");

		pushToMainBuffer("ergoishergoshe rgghsehrg hrhrehrg \nhrigergoo4iheorg    \nwefjo");
		pushToMainBuffer("WEFaerj a eorgis hrgs\nerg e\ne rgesrg serg\n sergserg");

	}

	void update(Input* input, Vec2 currentRes, float dt) {

		// Console variables.

		float consoleSpeed = 10;

		float bodyFontHeightPadding = 1.0f;
		float bodyFontHeightResultPadding = 1.2f;
		float bodyFontSize = 22;
		Font* bodyFont = getFont(FONT_SOURCESANS_PRO, bodyFontSize);
		float inputFontSize = 16;
		Font* inputFont = getFont(FONT_CONSOLAS, inputFontSize);

		float inputHeightPadding = 1.5;
		float fontDrawHeightOffset = 0.2f;
		Vec2 consolePadding = vec2(10,10);
		float cursorWidth = 2;
		float cursorSpeed = 3;
		float cursorColorMod = 0.2f;

		float g = 0.1f;
		Vec4 mouseHoverColorMod 	= vec4(g,g,g,0);
		Vec4 bodyColor 				= vec4(0.3f,0,0.6f,1.0f);
		Vec4 inputColor 			= vec4(0.37f,0,0.6f,1.0f);
		Vec4 bodyFontColor 			= vec4(1,1,1,1);
		Vec4 bodyFontResultColor 	= vec4(0.9f,0,0.1f,1);
		Vec4 inputFontColor 		= vec4(1,1,1,1);
		Vec4 cursorColor 			= vec4(0.2f,0.8f,0.1f,0.9f);
		Vec4 selectionColor 		= vec4(0.1f,0.4f,0.4f,0.9f);
		Vec4 scrollBarBackgroundColor = bodyColor + vec4(0.2,0,0,0);
		Vec4 scrollBarColor = vec4(0,0.5f,0.7f,1);

		float scrollBarWidth = 20;
		float scrollCursorMinHeight = 60;
		float scrollCursorMod = 0.3f;
		float scrollWheelAmount = bodyFontSize;
		float scrollWheelAmountMod = 4;

		// Logic.

		float closedPos = 0;
		float smallPos = -currentRes.y * CONSOLE_SMALL_PERCENT;
		float bigPos = -currentRes.y * CONSOLE_BIG_PERCENT;

		if(input->keysPressed[KEYCODE_F6]) {
			if(input->keysDown[KEYCODE_CTRL]) {
				if(mode == 0) mode = 2;
				else if(mode == 1) mode = 2;
				else mode = 0;
			} else {
				if(mode == 0) mode = 1;
				else if(mode == 2) mode = 1;
				else mode = 0;
			}
		}

		if(mode == 0) targetPos = 0;
		else if(mode == 1) targetPos = smallPos;
		else if(mode == 2) targetPos = bigPos;

		// Calculate smoothstep.

		float distance = pos - targetPos;
		if(abs(distance) > 0) {
			float consoleMovement = consoleSpeed * dt;
			pos = lerp(consoleMovement, pos, targetPos);

			// Clamp if we overstepped the target position.
			float newDistance = pos - targetPos;
			if(abs(newDistance) <= 1.0f) pos = targetPos;
		}



		Vec2 res = currentRes;
		float consoleTotalHeight = abs(pos);
		clampMin(&consoleTotalHeight, abs(smallPos));

		float inputHeight = inputFontSize * inputHeightPadding;
		float bodyTextHeight = bodyFontSize * bodyFontHeightPadding;
		float consoleBodyHeight = consoleTotalHeight - inputHeight;
		Rect consoleBody = rectMinDim(vec2(0, pos + inputHeight), vec2(res.w, consoleBodyHeight));
		Rect consoleInput = rectMinDim(vec2(0, pos), vec2(res.w, inputHeight));


		if(!isActive && pointInRect(input->mousePosNegative, consoleInput)) {
			inputColor += mouseHoverColorMod;

			if(input->mouseButtonPressed[0]) {
				isActive = true;
				strClear(inputBuffer);
				cursorIndex = 0;
				markerIndex = 0;
			}
		}

		if(input->keysPressed[KEYCODE_ESCAPE]) {
			isActive = false;
		}

		bool visible = true;
		if(pos == closedPos) {
			isActive = false;
			visible = false;
		}

		if(visible) {
			dcRect(consoleBody, bodyColor);
			dcRect(consoleInput, inputColor);


			float scrollOffset = 0;
			if(lastDiff >= 0) {
				scrollOffset = scrollPercent*lastDiff; 
				
				// Scrollbar background.

				Rect scrollRect = consoleBody;
				scrollRect.min.x = scrollRect.max.x - scrollBarWidth;

				// Scrollbar cursor.

				float consoleHeight = rectGetDim(consoleBody).h;
				float scrollCursorHeight = (consoleHeight / (consoleHeight + lastDiff)) * consoleHeight;
				clampMin(&scrollCursorHeight, scrollCursorMinHeight);


				if(input->mouseWheel && pointInRect(input->mousePosNegative, consoleBody)) {
					if(input->keysDown[KEYCODE_CTRL]) scrollWheelAmount *= scrollWheelAmountMod;

					scrollPercent += -input->mouseWheel * (scrollWheelAmount / lastDiff);
					clamp(&scrollPercent, 0, 1);
				}

				// Enable scrollbar keyboard navigation if we're not typing anything into the input console.
				if(strLen(inputBuffer) == 0) {
					if(input->keysPressed[KEYCODE_HOME]) scrollPercent = 0;
					if(input->keysPressed[KEYCODE_END]) scrollPercent = 1;
					if(input->keysPressed[KEYCODE_PAGEUP] || input->keysPressed[KEYCODE_PAGEDOWN]) {
						float dir;
						if(input->keysPressed[KEYCODE_PAGEUP]) dir = -1;
						else dir = 1;

						scrollPercent += dir * (consoleHeight*0.8f / lastDiff);
					}

					clamp(&scrollPercent, 0, 1);
				}

				// @CodeDuplication.

				float scrollCursorPos = lerp(scrollPercent, scrollRect.max.y - scrollCursorHeight/2, scrollRect.min.y + scrollCursorHeight/2);
				Rect scrollCursorRect = rectCenDim(vec2(scrollRect.max.x - scrollBarWidth/2, scrollCursorPos), vec2(scrollBarWidth,scrollCursorHeight));


				if(input->mouseButtonReleased[0]) {
					scrollMode = false;
				}

				if(pointInRect(input->mousePosNegative, scrollCursorRect)) {
					if(input->mouseButtonPressed[0]) {
						scrollMode = true;
						mouseAnchor = scrollCursorPos - input->mousePosNegative.y;
					}

					if(!scrollMode) {
						scrollBarColor += mouseHoverColorMod;
					}
				}

				if(scrollMode) {
					scrollBarColor += mouseHoverColorMod;

					scrollPercent = mapRangeClamp(input->mousePosNegative.y + mouseAnchor, scrollRect.min.y + scrollCursorHeight/2, scrollRect.max.y - scrollCursorHeight/2, 0, 1);
					scrollPercent = 1-scrollPercent;
				}

				// @CodeDuplication.
				// Recalculate scroll rect to reduce lag.

				scrollCursorPos = lerp(scrollPercent, scrollRect.max.y - scrollCursorHeight/2, scrollRect.min.y + scrollCursorHeight/2);
				scrollCursorRect = rectCenDim(vec2(scrollRect.max.x - scrollBarWidth/2, scrollCursorPos), vec2(scrollBarWidth,scrollCursorHeight));

				// Draw scrollbar.

				dcRect(scrollRect, scrollBarBackgroundColor);
				dcRect(scrollCursorRect, scrollBarColor);
			}

			if(mainBufferSize > 0) {
				dcEnable(STATE_SCISSOR);

				Rect scrollRect = consoleBody;
				scrollRect.min.x = scrollRect.max.x - scrollBarWidth;

				Rect consoleTextRect = consoleBody;
				consoleTextRect.max.x -= consolePadding.x;
				if(lastDiff >= 0) consoleTextRect.max.x -= scrollBarWidth;

				dcScissor(scissorRectScreenSpace(consoleTextRect, res.h));

				char* pre = "> ";
				float preSize = getTextDim(pre, bodyFont).w;

				Vec2 textPos = vec2(consolePadding.x + preSize, pos + consoleTotalHeight + scrollOffset - consolePadding.y);
				float textStart = textPos.y;
				// bool mousePressed = pointInRect(input->mousePosNegative, consoleTextRect) ? true : false;
				float textStartX = textPos.x;
				float wrappingWidth = rectGetDim(consoleTextRect).w - textStartX;

				bool mousePressed = input->mouseButtonPressed[0];
				bool mouseInsideConsole = pointInRect(input->mousePosNegative, consoleTextRect);
				bool mouseInsideScrollbar = pointInRect(input->mousePosNegative, scrollRect);
				if(mousePressed) {
					if(mouseInsideConsole) bodySelectionMode = true;
					else if(mouseInsideScrollbar) bodySelectionMode = false;
					else {
						bodySelectionMode = false;
						bodySelectionIndex = -1;
					}
				}

				if(input->mouseButtonReleased[0]) {
					bodySelectionMode = false;

					if(bodySelectionMarker1 == bodySelectionMarker2) {
						bodySelectionIndex = -1;
					}
				}

				for(int i = 0; i < mainBufferSize; i++) {
					if(i%2 == 0) {
						dcText(pre, bodyFont, textPos - vec2(preSize,0), bodyFontColor, 0, 2);
					} else {
						if(strIsEmpty(mainBuffer[i])) continue;
					}

					// Cull texts that are above or below the console body.
					int textHeight = getTextHeightWithWrapping(mainBuffer[i], bodyFont, textPos, wrappingWidth);
					bool textOutside = textPos.y - textHeight > consoleTextRect.max.y || textPos.y < consoleTextRect.min.y;

					if(!textOutside) {
						if(mousePressed && mouseInsideConsole) {
							if(valueBetween(input->mousePosNegative.y, textPos.y - textHeight, textPos.y) && 
							   (input->mousePosNegative.x < consoleTextRect.max.x)) {
								bodySelectionIndex = i;
								bodySelectionMarker1 = getTextPosWrapping(mainBuffer[bodySelectionIndex], bodyFont, textPos, input->mousePosNegative, wrappingWidth);
								bodySelectionMarker2 = bodySelectionMarker1;
								mousePressed = false;
							} 
						}

						if(i == bodySelectionIndex) {
							if(bodySelectionMode) {
								bodySelectionMarker2 = getTextPosWrapping(mainBuffer[bodySelectionIndex], bodyFont, textPos, input->mousePosNegative, wrappingWidth);
							}

							drawTextSelection(mainBuffer[i], bodyFont, textPos, bodySelectionMarker1, bodySelectionMarker2, selectionColor, wrappingWidth);
						}

						Vec4 color = i%2 == 0 ? bodyFontColor : bodyFontResultColor;
						dcText(mainBuffer[i], bodyFont, textPos, color, 0, 2, 0, vec4(0,0,0,0), wrappingWidth);
					}

					textPos.y -= textHeight;
				}

				lastDiff = textStart - textPos.y - rectGetDim(consoleTextRect).h + consolePadding.y*2;

				if(cursorIndex != markerIndex) {
					bodySelectionIndex = -1;
				}

				if(bodySelectionIndex != 0) {
					if(input->keysDown[KEYCODE_CTRL] && input->keysPressed[KEYCODE_C]) {
						char* selection = textSelectionToString(mainBuffer[bodySelectionIndex], bodySelectionMarker1, bodySelectionMarker2);
						setClipboard(selection);
					}
				}

				dcDisable(STATE_SCISSOR);
			}
		}

		if(isActive) {
			if(input->mouseButtonPressed[0]) {
				if(pointInRect(input->mousePosNegative, consoleInput)) {
					mouseSelectMode = true;
				} else {
					markerIndex = cursorIndex;
				}
			}

			if(input->mouseButtonReleased[0]) {
				mouseSelectMode = false;
			}

			if(mouseSelectMode && (strLen(inputBuffer) >= 1)) {
				float inputMid = pos + inputHeight/2;
				int mouseIndex = getTextPosWrapping(inputBuffer, inputFont, vec2(consolePadding.x, inputMid + inputFont->height/2), input->mousePosNegative);

				if(input->mouseButtonPressed[0]) {
					markerIndex = mouseIndex;
				}

				cursorIndex = mouseIndex;
			}

			if(!mouseSelectMode) {
				bool left = input->keysPressed[KEYCODE_LEFT];
				bool right = input->keysPressed[KEYCODE_RIGHT];
				bool up = input->keysPressed[KEYCODE_UP];
				bool down = input->keysPressed[KEYCODE_DOWN];

				bool a = input->keysPressed[KEYCODE_A];
				bool x = input->keysPressed[KEYCODE_X];
				bool c = input->keysPressed[KEYCODE_C];
				bool v = input->keysPressed[KEYCODE_V];
				
				bool home = input->keysPressed[KEYCODE_HOME];
				bool end = input->keysPressed[KEYCODE_END];
				bool backspace = input->keysPressed[KEYCODE_BACKSPACE];
				bool del = input->keysPressed[KEYCODE_DEL];
				bool enter = input->keysPressed[KEYCODE_RETURN];

				bool ctrl = input->keysDown[KEYCODE_CTRL];
				bool shift = input->keysDown[KEYCODE_SHIFT];


				int startCursorIndex = cursorIndex;

				if(ctrl && backspace) {
					shift = true;
					left = true;
				}

				if(ctrl && del) {
					shift = true;
					right = true;
				}

				if(left) {
					if(ctrl) {
						if(cursorIndex > 0) {
							while(inputBuffer[cursorIndex-1] == ' ') cursorIndex--;

							if(cursorIndex > 0)
						 		cursorIndex = strFindBackwards(inputBuffer, ' ', cursorIndex-1);
						}
					} else {
						bool isSelected = cursorIndex != markerIndex;
						if(isSelected && !shift) {
							if(cursorIndex < markerIndex) {
								markerIndex = cursorIndex;
							} else {
								cursorIndex = markerIndex;
							}
						} else {
							if(cursorIndex > 0) cursorIndex--;
						}
					}
				}

				if(right) {
					if(ctrl) {
						while(inputBuffer[cursorIndex] == ' ') cursorIndex++;
						if(cursorIndex <= strLen(inputBuffer)) {
							cursorIndex = strFindOrEnd(inputBuffer, ' ', cursorIndex+1);
							if(cursorIndex != strLen(inputBuffer)) cursorIndex--;
						}
					} else {
						bool isSelected = cursorIndex != markerIndex;
						if(isSelected && !shift) {
							if(cursorIndex > markerIndex) {
								markerIndex = cursorIndex;
							} else {
								cursorIndex = markerIndex;
							}
						} else {
							if(cursorIndex < strLen(inputBuffer)) cursorIndex++;
						}
					}
				}

				if(home) {
					cursorIndex = 0;
				}

				if(end) {
					cursorIndex = strLen(inputBuffer);
				}

				if((startCursorIndex != cursorIndex) && !shift) {
					markerIndex = cursorIndex;
				}

				if(ctrl && a) {
					cursorIndex = 0;
					markerIndex = strLen(inputBuffer);
				}

				bool isSelected = cursorIndex != markerIndex;

				if(ctrl && x) {
					c = true;
					del = true;
				}

				if(ctrl && c) {
					if(cursorIndex != markerIndex) {
						char* selection = textSelectionToString(inputBuffer, cursorIndex, markerIndex);
						setClipboard(selection);
					}
				}

				if(backspace || del || (input->inputCharacterCount > 0) || (ctrl && v)) {
					if(isSelected) {
						int delIndex = min(cursorIndex, markerIndex);
						int delAmount = abs(cursorIndex - markerIndex);
						strRemoveX(inputBuffer, delIndex, delAmount);
						cursorIndex = delIndex;
					}

					markerIndex = cursorIndex;
				}

				if(ctrl && v) {
					char* clipboard = (char*)getClipboard();
					int clipboardSize = strLen(clipboard);
					if(clipboardSize + strLen(inputBuffer) < arrayCount(inputBuffer)) {
						strInsert(inputBuffer, cursorIndex, clipboard);
						cursorIndex += clipboardSize;
						markerIndex = cursorIndex;
					}
				}

				// Add input characters to input buffer.
				if(input->inputCharacterCount > 0) {
					if(input->inputCharacterCount + strLen(inputBuffer) < arrayCount(inputBuffer)) {
						strInsert(inputBuffer, cursorIndex, input->inputCharacters, input->inputCharacterCount);
						cursorIndex += input->inputCharacterCount;
						markerIndex = cursorIndex;
					}
				}

				if(backspace && !isSelected) {
					if(cursorIndex > 0) {
						strRemove(inputBuffer, cursorIndex);
						cursorIndex--;
					}
					markerIndex = cursorIndex;
				}

				if(del && !isSelected) {
					if(cursorIndex+1 <= strLen(inputBuffer)) {
						strRemove(inputBuffer, cursorIndex+1);
					}
					markerIndex = cursorIndex;
				}

				if(enter) {
					if(strLen(inputBuffer) > 0) {
						// Copy over input buffer to console buffer.
						pushToMainBuffer(inputBuffer);
						evaluateInput();

						strClear(inputBuffer);
						cursorIndex = 0;
						markerIndex = 0;

						scrollPercent = 1;
					}
				}

				if(startCursorIndex != cursorIndex) {
					cursorTime = 0;
				}
			}

			// 
			// Drawing.
			//
			
			float inputMid = pos + inputHeight/2;

			// Selection.

			drawTextSelection(inputBuffer, inputFont, vec2(consolePadding.x, inputMid + inputFontSize/2), cursorIndex, markerIndex, selectionColor);

			// Text.

			dcText(inputBuffer, inputFont, vec2(consolePadding.x, consoleInput.min.y + inputHeight/2 + inputFontSize * fontDrawHeightOffset), inputFontColor, 0, 1);

			// Cursor.

			cursorTime += dt*cursorSpeed;
			Vec4 cmod = vec4(0,cos(cursorTime)*cursorColorMod - cursorColorMod,0,0);

			Vec2 cursorPos = getTextMousePos(inputBuffer, inputFont, vec2(consolePadding.x, inputMid + inputFontSize/2), cursorIndex);
			float cWidth = cursorIndex == strLen(inputBuffer) ? getTextDim("M", inputFont).w : cursorWidth;
			Rect cursorRect = rectULDim(cursorPos, vec2(cWidth, inputFontSize));

			dcRect(cursorRect, cursorColor + cmod);
		}
	}

	char* eatWhiteSpace(char* str) {
		int index = 0;
		while(str[index] == ' ') index++;
		return str + index;
	}

	char* eatSign(char* str) {
		char c = str[0];
		if(c == '-' || c == '+') return str + 1;
		else return str;
	}

	bool charIsDigit(char c) {
		return (c >= (int)'0') && (c <= (int)'9');
	}

	char* eatDigits(char* str) {
		while(charIsDigit(str[0])) str++;
		return str;
	}

	char* getNextArgument(char** s) {
		*s = eatWhiteSpace(*s);
		char* str = *s;

		if(str[0] == '\0') return 0;
		int wpos = strFindX(str, ' ');
		if(wpos == -1) wpos = strLen(str) + 1;
		wpos--;

		char* argument = getTStringDebug(wpos + 1);
		strCpy(argument, str, wpos);

		*s += wpos;

		return argument;
	}

	void pushToMainBuffer(char* str) {
		char* newString = getPArrayDebug(char, strLen(str) + 1);
		strCpy(newString, str);

		mainBuffer[mainBufferSize] = newString;
		mainBufferSize++;
	}

	bool strIsInteger(char* str) {
		char* s = str;
		s = eatSign(s);
		if(!charIsDigit(s[0])) return false;
		s = eatDigits(s);

		if(s[0] != '\0') return false;

		return true;
	}

	void evaluateInput() {
		char* com = inputBuffer;

		char* argument = getNextArgument(&com);
		if(argument == 0) return;

		if(strCompare(argument, "add")) {
			char* arguments[2];
			arguments[0] = getNextArgument(&com);
			arguments[1] = getNextArgument(&com);
			if(!(arguments + 0) || !(arguments + 1)) {
				pushToMainBuffer(fillString("Function is missing arguments."));
				return;
			}

			if(!strIsInteger(arguments[0]) || !strIsInteger(arguments[1])) {
				pushToMainBuffer(fillString("Arguments are not integers."));
				return;
			}

			int a = strToInt(arguments[0]);
			int b = strToInt(arguments[1]);

			pushToMainBuffer(fillString("%i + %i = %i", a, b, a+b));
		} else if(strCompare(argument, "donothing")) {
			pushToMainBuffer("");
		} else {
			pushToMainBuffer(fillString("Unknown command \"%s\"", argument));
		}
	}
};




struct DebugState {
	bool showHud;
	bool showConsole;

	DrawCommandList commandListDebug;
	Input* input;

	bool isInitialised;
	int timerInfoCount;
	TimerInfo timerInfos[32]; // timerInfoCount
	TimerSlot* timerBuffer;
	int bufferSize;
	u64 bufferIndex;

	Timings timings[120][32];
	int cycleIndex;

	bool frozenGraph;
	TimerSlot* savedTimerBuffer;
	u64 savedBufferIndex;
	Timings savedTimings[32];

	GuiInput gInput;
	Gui* gui;
	Gui* gui2;

	Input* recordedInput;
	int inputCapacity;
	bool recordingInput;
	int inputIndex;

	bool playbackStart;
	bool playbackInput;
	int playbackIndex;
	bool playbackSwapMemory;

	// Console.

	Console console;
};




extern DebugState* globalDebugState;

inline uint getThreadID() {
	char *threadLocalStorage = (char *)__readgsqword(0x30);
	uint threadID = *(uint *)(threadLocalStorage + 0x48);

	return threadID;
}

void addTimerSlot(int timerIndex, int type) {
	// uint id = atomicAdd64(&globalDebugState->bufferIndex, 1);
	uint id = globalDebugState->bufferIndex++;
	TimerSlot* slot = globalDebugState->timerBuffer + id;
	slot->cycles = __rdtsc();
	slot->type = type;
	slot->threadId = getThreadID();
	slot->timerIndex = timerIndex;
}

void addTimerSlotAndInfo(int timerIndex, int type, const char* file, const char* function, int line, char* name = "") {

	TimerInfo* timerInfo = globalDebugState->timerInfos + timerIndex;

	if(!timerInfo->initialised) {
		timerInfo->initialised = true;
		timerInfo->file = file;
		timerInfo->function = function;
		timerInfo->line = line;
		timerInfo->type = type;
		timerInfo->name = name;
	}

	addTimerSlot(timerIndex, type);
}

struct TimerBlock {
	int counter;

	TimerBlock(int counter, const char* file, const char* function, int line, char* name = "") {

		this->counter = counter;
		addTimerSlotAndInfo(counter, TIMER_TYPE_BEGIN, file, function, line, name);
	}

	~TimerBlock() {
		addTimerSlot(counter, TIMER_TYPE_END);
	}
};

#define TIMER_BLOCK() \
	TimerBlock timerBlock##__LINE__(__COUNTER__, __FILE__, __FUNCTION__, __LINE__);

#define TIMER_BLOCK_NAMED(name) \
	TimerBlock timerBlock##__LINE__(__COUNTER__, __FILE__, __FUNCTION__, __LINE__, name);

#define TIMER_BLOCK_BEGIN(ID) \
	const int timerCounter##ID = __COUNTER__; \
	addTimerSlotAndInfo(timerCounter##ID, TIMER_TYPE_BEGIN, __FILE__, __FUNCTION__, __LINE__); 

#define TIMER_BLOCK_BEGIN_NAMED(ID, name) \
	const int timerCounter##ID = __COUNTER__; \
	addTimerSlotAndInfo(timerCounter##ID, TIMER_TYPE_BEGIN, __FILE__, __FUNCTION__, __LINE__, name); 

#define TIMER_BLOCK_END(ID) \
	addTimerSlot(timerCounter##ID, TIMER_TYPE_END);

struct Statistic {
	f64 min;
	f64 max;
	f64 avg;
	int count;
};

void beginStatistic(Statistic* stat) {
	stat->min = DBL_MAX; 
	stat->max = -DBL_MAX; 
	stat->avg = 0; 
	stat->count = 0; 
}

void updateStatistic(Statistic* stat, f64 value) {
	if(value < stat->min) stat->min = value;
	if(value > stat->max) stat->max = value;
	stat->avg += value;
	++stat->count;
}

void endStatistic(Statistic* stat) {
	stat->avg /= stat->count;
}

struct TimerStatistic {
	u64 cycles;
	int timerIndex;
};


